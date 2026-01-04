from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from golfcoach.io.npz_io import load_npz
from golfcoach.io.rig_config import load_rig_config
from golfcoach.pose3d.joints3d_kf_triang_opt import undistort_kpts_pixels


def _print_npz_keys(label: str, npz_path: str) -> None:
	data = np.load(npz_path, allow_pickle=True)
	print(f"[{label}] {npz_path} keys:", list(data.files))


def _safe_mkdir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def _hashable_size(x: np.ndarray | Sequence[int]) -> Tuple[int, int]:
	a = np.asarray(x).ravel()
	if a.size >= 2:
		return int(a[0]), int(a[1])
	return 0, 0


def _extract_xy_conf_from_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Tuple[int, int]]:
	"""
	Returns (xy[T,J,2], conf[T,J], frame_paths or None, image_size(W,H))
	Heuristics over common key names / structures.
	"""
	d = np.load(npz_path, allow_pickle=True)
	keys = set(d.files)
	# Image size
	image_size = (0, 0)
	for k in ["image_size", "img_size", "size", "resolution"]:
		if k in keys:
			image_size = _hashable_size(d[k])
			break

	xy: Optional[np.ndarray] = None
	conf: Optional[np.ndarray] = None
	frame_paths: Optional[List[str]] = None

	# Common direct arrays
	candidate_xy = [
		"keypoints", "keypoints2d", "joints2d", "kpts", "kp", "pose2d", "kps",
	]
	for k in candidate_xy:
		if k in keys:
			arr = np.asarray(d[k])
			if arr.ndim == 3 and arr.shape[-1] in (2, 3):
				if arr.shape[-1] == 3:
					xy = arr[..., :2].astype(np.float32)
					conf = arr[..., 2].astype(np.float32)
				else:
					xy = arr.astype(np.float32)
				break

	# Confidence separate?
	if conf is None:
		for k in ["conf", "confidence", "scores", "score"]:
			if k in keys:
				conf = np.asarray(d[k]).astype(np.float32)
				break

	# If still missing, check for per-frame object arrays
	if xy is None or conf is None:
		for k in ["outputs", "frames"]:
			if k in keys:
				obj = d[k]
				if isinstance(obj, np.ndarray) and obj.dtype == object:
					T = len(obj)
					first = obj[0]
					# Try keys inside each frame dict-like
					cand = ["keypoints", "keypoints2d", "joints2d", "kpts", "pose2d"]
					xy_key = None
					for ck in cand:
						if isinstance(first, dict) and ck in first:
							xy_key = ck
							break
					if xy_key is not None:
						frames_xy: List[np.ndarray] = []
						frames_conf: List[np.ndarray] = []
						for t in range(T):
							entry = obj[t]
							arr = np.asarray(entry[xy_key])
							if arr.ndim == 2 and arr.shape[1] >= 2:
								xy2 = arr[:, :2].astype(np.float32)
								frames_xy.append(xy2)
								if arr.shape[1] >= 3:
									frames_conf.append(arr[:, 2].astype(np.float32))
								else:
									frames_conf.append(np.ones((xy2.shape[0],), dtype=np.float32))
							else:
								raise ValueError(f"Unexpected per-frame {xy_key} shape: {arr.shape}")
						xy = np.stack(frames_xy, axis=0)
						conf = np.stack(frames_conf, axis=0)
				break

	# Frame paths if present
	for k in ["frame_paths", "paths", "images", "image_paths"]:
		if k in keys:
			arr = np.asarray(d[k]).ravel().tolist()
			frame_paths = [str(x) for x in arr]
			break

	if xy is None or conf is None:
		raise ValueError(f"Failed to extract xy/conf from {npz_path}")

	# If confidence absent earlier, default to ones
	if conf is None:
		conf = np.ones(xy.shape[:2], dtype=np.float32)

	return xy.astype(np.float32), conf.astype(np.float32), frame_paths, image_size


def _raw_to_used_batch(xy: np.ndarray, K: np.ndarray, D: np.ndarray, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
	"""
	Apply cv2.undistortPoints with P=K in pixel space to batch (T,J,2) points.
	Assumes intrinsics have already been scaled to dst_size if needed.
	"""
	T, J, _ = xy.shape
	pts = xy.reshape(T * J, 1, 2).astype(np.float64)
	und = cv2.undistortPoints(pts, K.astype(np.float64), D.astype(np.float64), P=K.astype(np.float64))
	return und.reshape(T, J, 2).astype(np.float32)


def _resolve_frame_image(frame_idx: int, fallback_size: Tuple[int, int], candidates: List[Path]) -> np.ndarray:
	for p in candidates:
		if p.exists():
			img = cv2.imread(str(p), cv2.IMREAD_COLOR)
			if img is not None:
				return img
	# Fallback: blank canvas
	W, H = int(fallback_size[0]), int(fallback_size[1])
	W = W if W > 0 else 1280
	H = H if H > 0 else 720
	canvas = np.zeros((H, W, 3), dtype=np.uint8)
	cv2.putText(canvas, f"frame {frame_idx} (no image)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 2, cv2.LINE_AA)
	return canvas


def _frame_path_candidates(base: Optional[Path], idx: int) -> List[Path]:
	if base is None:
		return []
	n = f"{idx:06d}"
	return [
		base / f"{n}.jpg",
		base / f"{n}.png",
		base / "img" / f"{n}.jpg",
		base / "img" / f"{n}.png",
	]


def _draw_circle(img: np.ndarray, pt: Tuple[float, float], color: Tuple[int, int, int], radius: int, thickness: int = -1) -> None:
	if pt is None or not np.all(np.isfinite(pt)):
		return
	x, y = int(round(pt[0])), int(round(pt[1]))
	cv2.circle(img, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)


def _draw_cross(img: np.ndarray, pt: Tuple[float, float], color: Tuple[int, int, int], size: int = 4, thickness: int = 1) -> None:
	if pt is None or not np.all(np.isfinite(pt)):
		return
	x, y = int(round(pt[0])), int(round(pt[1]))
	cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
	cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)


def _radius_from_conf(c: float, r_min: int = 2, r_max: int = 10, gamma: float = 2.0) -> int:
	c = float(np.clip(c, 0.0, 1.0))
	val = r_min + (c ** gamma) * (r_max - r_min)
	return int(round(val))


def _overlay2d_video(out_path: Path, xy: np.ndarray, conf: np.ndarray, valid: np.ndarray, rejected: np.ndarray,
                     frame_idx: np.ndarray, image_size: Tuple[int, int],
                     frame_paths: Optional[List[str]], frames_dir: Optional[Path], fps: int, legend: str) -> None:
	_safe_mkdir(out_path.parent)
	W, H = int(image_size[0]), int(image_size[1])
	W = W if W > 0 else 1280
	H = H if H > 0 else 720
	writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
	T, J, _ = xy.shape
	for ti in range(T):
		if frame_paths and ti < len(frame_paths) and frame_paths[ti]:
			img = cv2.imread(frame_paths[ti], cv2.IMREAD_COLOR)
			if img is None:
				img = _resolve_frame_image(int(frame_idx[ti]), image_size, _frame_path_candidates(frames_dir, int(frame_idx[ti])) if frames_dir else [])
		else:
			img = _resolve_frame_image(int(frame_idx[ti]), image_size, _frame_path_candidates(frames_dir, int(frame_idx[ti])) if frames_dir else [])
		# Frame counter (top-left)
		cnt_text = f"{ti + 1}/{T}  f:{int(frame_idx[ti])}"
		cv2.putText(img, cnt_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
		cv2.putText(img, cnt_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
		for j in range(J):
			pt = tuple(xy[ti, j].tolist())
			c = float(conf[ti, j])
			r = _radius_from_conf(c)
			if rejected[ti, j]:
				_draw_cross(img, pt, (0, 128, 255), size=6, thickness=2)
			elif valid[ti, j]:
				_draw_circle(img, pt, (0, 255, 0), r)
			else:
				_draw_circle(img, pt, (128, 128, 128), max(2, r // 2), thickness=1)
		cv2.putText(img, legend, (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
		writer.write(img)
	writer.release()
	print(f"[viz] wrote {out_path}")


def _overlay_reproj_video(out_path: Path, u_obs: np.ndarray, reproj: np.ndarray, valid: np.ndarray, rejected: np.ndarray,
                          frame_idx: np.ndarray, image_size: Tuple[int, int], frame_paths: Optional[List[str]], frames_dir: Optional[Path],
                          fps: int, err_thr: float, title: str) -> None:
	_safe_mkdir(out_path.parent)
	W, H = int(image_size[0]), int(image_size[1])
	W = W if W > 0 else 1280
	H = H if H > 0 else 720
	writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
	T, J, _ = u_obs.shape
	for ti in range(T):
		if frame_paths and ti < len(frame_paths) and frame_paths[ti]:
			img = cv2.imread(frame_paths[ti], cv2.IMREAD_COLOR)
			if img is None:
				img = _resolve_frame_image(int(frame_idx[ti]), image_size, _frame_path_candidates(frames_dir, int(frame_idx[ti])) if frames_dir else [])
		else:
			img = _resolve_frame_image(int(frame_idx[ti]), image_size, _frame_path_candidates(frames_dir, int(frame_idx[ti])) if frames_dir else [])
		# Frame counter (top-left)
		cnt_text = f"{ti + 1}/{T}  f:{int(frame_idx[ti])}"
		cv2.putText(img, cnt_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
		cv2.putText(img, cnt_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
		for j in range(J):
			pt_obs = u_obs[ti, j]
			pt_rep = reproj[ti, j]
			fin_obs = np.all(np.isfinite(pt_obs))
			fin_rep = np.all(np.isfinite(pt_rep))

			# Skip if both invalid
			if not fin_obs and not fin_rep:
				continue

			# If observed invalid, draw reproj (if finite) in gray and continue
			if not fin_obs:
				if fin_rep:
					_draw_cross(img, tuple(pt_rep.tolist()), (128, 128, 128), size=4, thickness=1)
				continue

			# If reproj invalid, draw observed in gray and continue
			if not fin_rep:
				_draw_circle(img, tuple(pt_obs.tolist()), (128, 128, 128), 4)
				continue

			# Both finite: compute error and draw
			e = float(np.linalg.norm(pt_obs - pt_rep))
			color = (0, 255, 0) if e < err_thr else (0, 0, 255)
			if rejected[ti, j]:
				color = (128, 128, 128)

			_draw_circle(img, tuple(pt_obs.tolist()), color, 4)
			_draw_cross(img, tuple(pt_rep.tolist()), color, size=5, thickness=2)
			cv2.line(img, (int(round(pt_obs[0])), int(round(pt_obs[1]))), (int(round(pt_rep[0])), int(round(pt_rep[1]))), color, 1, cv2.LINE_AA)
			if e >= err_thr:
				cv2.putText(img, f"{e:.1f}", (int(round(pt_rep[0])) + 5, int(round(pt_rep[1])) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
		cv2.putText(img, title, (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
		writer.write(img)
	writer.release()
	print(f"[viz] wrote {out_path}")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--pose2d_left", required=True)
	ap.add_argument("--pose2d_right", required=True)
	ap.add_argument("--stageb_npz", required=True)
	ap.add_argument("--rig_json", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--t0", type=int, default=0)
	ap.add_argument("--t1", type=int, default=-1)
	ap.add_argument("--fps", type=int, default=120)
	ap.add_argument("--err_thr", type=float, default=30.0)
	ap.add_argument("--mode", choices=["2d", "reproj", "3d", "all"], default="all")
	ap.add_argument("--left_frames_dir", default=None)
	ap.add_argument("--right_frames_dir", default=None)
	ap.add_argument("--hide_club", action="store_true", help="Do not render club joints/edges in 3D viewer")
	ap.add_argument("--all_green_3d", action="store_true", help="Render all joints in green (no outlier coloring) in 3D viewer")
	ap.add_argument("--labels_3d", action="store_true", help="Show joint name labels in 3D viewer")
	args = ap.parse_args()

	out_dir = Path(args.out_dir)
	_safe_mkdir(out_dir)

	# Print keys
	_print_npz_keys("pose2d_left", args.pose2d_left)
	_print_npz_keys("pose2d_right", args.pose2d_right)
	_print_npz_keys("stageB", args.stageb_npz)

	# Load StageB
	sb = np.load(args.stageb_npz, allow_pickle=True)
	j3d = np.asarray(sb["joints3d"]).astype(np.float32)
	frame_idx = np.asarray(sb["frame_idx"]).astype(np.int32)
	joint_names = [str(x) for x in np.asarray(sb["joint_names"]).tolist()]
	valid2d_left = np.asarray(sb["valid2d_left"]).astype(bool)
	valid2d_right = np.asarray(sb["valid2d_right"]).astype(bool)
	rej2d_left = np.asarray(sb["rejected2d_left"]).astype(bool)
	rej2d_right = np.asarray(sb["rejected2d_right"]).astype(bool)
	reproj_opt_left = np.asarray(sb["reproj_opt_left"]).astype(np.float32)
	reproj_opt_right = np.asarray(sb["reproj_opt_right"]).astype(np.float32)
	reproj_init_left = sb["reproj_init_left"].astype(np.float32) if "reproj_init_left" in sb.files else None
	reproj_init_right = sb["reproj_init_right"].astype(np.float32) if "reproj_init_right" in sb.files else None
	T, J, _ = j3d.shape

	# Load rig
	img_sz_rig, K0, D0, K1, D1, R, Tvec = load_rig_config(args.rig_json)

	# Load 2D pose and convert to "used/refined" (undistorted pixel space with P=K)
	xyL_raw, confL, frame_paths_L, img_size_L = _extract_xy_conf_from_npz(args.pose2d_left)
	xyR_raw, confR, frame_paths_R, img_size_R = _extract_xy_conf_from_npz(args.pose2d_right)
	# Scale intrinsics if needed
	Wc, Hc = int(img_sz_rig[0]), int(img_sz_rig[1])
	Wn, Hn = int(img_size_L[0]), int(img_size_L[1])
	if (Wc, Hc) != (Wn, Hn):
		sx = float(Wn) / float(Wc) if Wc > 0 else 1.0
		sy = float(Hn) / float(Hc) if Hc > 0 else 1.0
		K0 = K0.copy().astype(np.float64)
		K1 = K1.copy().astype(np.float64)
		K0[0, 0] *= sx; K0[0, 2] *= sx; K0[1, 1] *= sy; K0[1, 2] *= sy
		K1[0, 0] *= sx; K1[0, 2] *= sx; K1[1, 1] *= sy; K1[1, 2] *= sy

	u_used_left = _raw_to_used_batch(xyL_raw, K0, D0, img_sz_rig, img_size_L)
	u_used_right = _raw_to_used_batch(xyR_raw, K1, D1, img_sz_rig, img_size_R)

	# Time slice
	t0 = max(0, int(args.t0))
	t1 = int(args.t1)
	if t1 < 0 or t1 > len(frame_idx):
		t1 = len(frame_idx)
	sl = slice(t0, t1)
	def S(x):
		return x[sl] if x is not None else None

	# Debug prints
	def _finite_pct(a: np.ndarray) -> float:
		return float(np.isfinite(a).all(axis=-1).mean()) * 100.0 if a is not None else 0.0
	print("[shapes] u_used_left/right:", u_used_left.shape, u_used_right.shape)
	print("[shapes] conf L/R:", confL.shape, confR.shape)
	print("[shapes] reproj opt L/R:", reproj_opt_left.shape, reproj_opt_right.shape)
	print("[finite %] u_used_left/right:", f"{_finite_pct(u_used_left):.1f}%", f"{_finite_pct(u_used_right):.1f}%")

	# Mode: 2D overlay
	if args.mode in ("2d", "all"):
		left_dir = Path(args.left_frames_dir) if args.left_frames_dir else None
		right_dir = Path(args.right_frames_dir) if args.right_frames_dir else None
		_overlay2d_video(
			out_dir / "overlay2d_left.mp4",
			S(u_used_left), S(confL), S(valid2d_left), S(rej2d_left),
			S(frame_idx), img_size_L, frame_paths_L[t0:t1] if frame_paths_L else None, left_dir, args.fps,
			"Refined 2D (left). Dot size = confidence; rejected marked",
		)
		_overlay2d_video(
			out_dir / "overlay2d_right.mp4",
			S(u_used_right), S(confR), S(valid2d_right), S(rej2d_right),
			S(frame_idx), img_size_R, frame_paths_R[t0:t1] if frame_paths_R else None, right_dir, args.fps,
			"Refined 2D (right). Dot size = confidence; rejected marked",
		)

	# Mode: reprojection overlays
	if args.mode in ("reproj", "all"):
		left_dir = Path(args.left_frames_dir) if args.left_frames_dir else None
		right_dir = Path(args.right_frames_dir) if args.right_frames_dir else None
		_overlay_reproj_video(
			out_dir / "reproj_left_opt.mp4",
			S(u_used_left), S(reproj_opt_left), S(valid2d_left), S(rej2d_left),
			S(frame_idx), img_size_L, frame_paths_L[t0:t1] if frame_paths_L else None, left_dir, args.fps, args.err_thr,
			"OBS vs REPROJ (OPT) - Left",
		)
		_overlay_reproj_video(
			out_dir / "reproj_right_opt.mp4",
			S(u_used_right), S(reproj_opt_right), S(valid2d_right), S(rej2d_right),
			S(frame_idx), img_size_R, frame_paths_R[t0:t1] if frame_paths_R else None, right_dir, args.fps, args.err_thr,
			"OBS vs REPROJ (OPT) - Right",
		)
		if reproj_init_left is not None and reproj_init_right is not None:
			_overlay_reproj_video(
				out_dir / "reproj_left_init.mp4",
				S(u_used_left), S(reproj_init_left), S(valid2d_left), S(rej2d_left),
				S(frame_idx), img_size_L, frame_paths_L[t0:t1] if frame_paths_L else None, left_dir, args.fps, args.err_thr,
				"OBS vs REPROJ (INIT) - Left",
			)
			_overlay_reproj_video(
				out_dir / "reproj_right_init.mp4",
				S(u_used_right), S(reproj_init_right), S(valid2d_right), S(rej2d_right),
				S(frame_idx), img_size_R, frame_paths_R[t0:t1] if frame_paths_R else None, right_dir, args.fps, args.err_thr,
				"OBS vs REPROJ (INIT) - Right",
			)

	# Debug summary at a frame (default 79 if in range)
	dbg_t = 79 if (79 >= t0 and 79 < t1) else t0
	if dbg_t >= t0 and dbg_t < t1:
		errL = np.linalg.norm(S(u_used_left)[dbg_t - t0] - S(reproj_opt_left)[dbg_t - t0], axis=-1)
		errR = np.linalg.norm(S(u_used_right)[dbg_t - t0] - S(reproj_opt_right)[dbg_t - t0], axis=-1)
		err = np.maximum(errL, errR)
		top_idx = np.argsort(-err)[:10]
		print(f"[dbg t={dbg_t}] top-10 reproj errors:")
		for j in top_idx:
			print(f"  {j:02d} {joint_names[j]:>16s}  e={err[j]:6.2f}  validL/R={bool(S(valid2d_left)[dbg_t - t0, j])}/{bool(S(valid2d_right)[dbg_t - t0, j])}  rejL/R={bool(S(rej2d_left)[dbg_t - t0, j])}/{bool(S(rej2d_right)[dbg_t - t0, j])}")

	# 3D visualization (optional; placeholder for full interactive viewer)
	if args.mode in ("3d", "all"):
		try:
			import open3d as o3d
			import time

			def find_joint_idx(names: Sequence[str], cands: Sequence[str]) -> Optional[int]:
				lower = [s.lower() for s in names]
				for c in cands:
					cl = c.lower()
					# exact first
					for i, nm in enumerate(lower):
						if nm == cl:
							return i
					# substring
					for i, nm in enumerate(lower):
						if cl in nm:
							return i
				return None

			def build_edges(names: Sequence[str]) -> Tuple[List[Tuple[int, int]], List[int]]:
				edges: List[Tuple[int, int]] = []
				club_inds: List[int] = []
				def add(a: Optional[int], b: Optional[int]):
					if a is not None and b is not None:
						edges.append((a, b))
				# torso
				root  = find_joint_idx(names, ["root"])
				spine = find_joint_idx(names, ["spine"])
				thrx  = find_joint_idx(names, ["thorax"])
				neck  = find_joint_idx(names, ["neck_base", "neck"])
				head  = find_joint_idx(names, ["head"])
				add(root, spine); add(spine, thrx); add(thrx, neck); add(neck, head)
				# left leg
				lhip = find_joint_idx(names, ["left_hip","l_hip"])
				lkne = find_joint_idx(names, ["left_knee","l_knee"])
				lank = find_joint_idx(names, ["left_foot","l_foot","left_ankle","l_ankle"])
				add(root, lhip); add(lhip, lkne); add(lkne, lank)
				# right leg
				rhip = find_joint_idx(names, ["right_hip","r_hip"])
				rkne = find_joint_idx(names, ["right_knee","r_knee"])
				rank = find_joint_idx(names, ["right_foot","r_foot","right_ankle","r_ankle"])
				add(root, rhip); add(rhip, rkne); add(rkne, rank)
				# left arm
				lsho = find_joint_idx(names, ["left_shoulder","l_shoulder"])
				lelb = find_joint_idx(names, ["left_elbow","l_elbow"])
				lwri = find_joint_idx(names, ["left_wrist","l_wrist"])
				add(thrx, lsho); add(lsho, lelb); add(lelb, lwri)
				# right arm
				rsho = find_joint_idx(names, ["right_shoulder","r_shoulder"])
				relb = find_joint_idx(names, ["right_elbow","r_elbow"])
				rwri = find_joint_idx(names, ["right_wrist","r_wrist"])
				add(thrx, rsho); add(rsho, relb); add(relb, rwri)
				# club (optional)
				shaft = find_joint_idx(names, ["shaft"])
				hosel = find_joint_idx(names, ["hosel"])
				heel  = find_joint_idx(names, ["heel"])
				toed  = find_joint_idx(names, ["toe_down"])
				toeu  = find_joint_idx(names, ["toe_up"])
				for idx in [shaft, hosel, heel, toed, toeu]:
					if idx is not None:
						club_inds.append(idx)
				add(shaft, hosel); add(hosel, heel); add(hosel, toed); add(hosel, toeu)
				edges = [e for e in edges if e[0] is not None and e[1] is not None]
				return edges, club_inds

			# Prepare data slice
			J3D_OPT = j3d[sl].astype(np.float32)
			J3D_INIT = None
			if "joints3d_init" in sb.files:
				J3D_INIT = np.asarray(sb["joints3d_init"])[sl].astype(np.float32)

			# Outlier mask using reprojection error and rejected masks
			uL = u_used_left[sl]; uR = u_used_right[sl]
			rL = reproj_opt_left[sl]; rR = reproj_opt_right[sl]
			mrejL = rej2d_left[sl]; mrejR = rej2d_right[sl]
			eL = np.linalg.norm(uL - rL, axis=-1)
			eR = np.linalg.norm(uR - rR, axis=-1)
			err = np.maximum(eL, eR)
			outlier = (err >= args.err_thr) | mrejL | mrejR  # (T',J)

			edges_all, club_inds = build_edges(joint_names)
			J_total = J3D_OPT.shape[1]
			keep_inds = list(range(J_total))
			if args.hide_club and len(club_inds) > 0:
				rm = set(club_inds)
				keep_inds = [i for i in keep_inds if i not in rm]
			old_to_new = {old: new for new, old in enumerate(keep_inds)}
			lines_list: List[Tuple[int, int]] = []
			for a, b in edges_all:
				if a in old_to_new and b in old_to_new:
					lines_list.append((old_to_new[a], old_to_new[b]))
			lines = np.array(lines_list, dtype=np.int32) if len(lines_list) > 0 else np.empty((0, 2), dtype=np.int32)

			# Open3D geometries
			Ts = J3D_OPT.shape[0]
			if Ts <= 0 or J3D_OPT.shape[1] <= 0:
				print("[viz] 3D viewer: empty sequence after slicing; skipping viewer.")
				return

			# Initial points (ensure non-empty)
			P0 = J3D_INIT[0] if (J3D_INIT is not None) else J3D_OPT[0]
			if len(keep_inds) != P0.shape[0]:
				P0 = P0[keep_inds]
			if P0.shape[0] == 0:
				P0 = np.zeros((1, 3), dtype=np.float32)
			# Names after filtering (for labels)
			names_kept = [joint_names[i] for i in keep_inds] if len(keep_inds) > 0 else joint_names
			pc = o3d.geometry.PointCloud()
			pc.points = o3d.utility.Vector3dVector(P0.astype(np.float64))
			pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (P0.shape[0], 1)))

			ls = None
			if lines.size > 0:
				ls = o3d.geometry.LineSet()
				ls.points = o3d.utility.Vector3dVector(P0.astype(np.float64))
				ls.lines = o3d.utility.Vector2iVector(lines)

			# Colors
			def colors_for_frame(mask: np.ndarray) -> np.ndarray:
				# All green if requested
				if args.all_green_3d:
					col = np.zeros((mask.shape[0], 3), dtype=np.float32)
					col[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
					return col
				# Default: green for inlier, red for outlier
				col = np.zeros((mask.shape[0], 3), dtype=np.float32)
				col[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
				col[mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
				return col

			# State
			state = {"t": 0, "playing": True, "use_init": False}

			# Labels helper
			def _update_labels(vis: o3d.visualization.Visualizer, P: np.ndarray) -> None:
				if not args.labels_3d:
					return
				# Cache capability check (some Open3D builds don't expose 3D label APIs)
				if not hasattr(_update_labels, "_support_checked"):
					_update_labels._support_checked = True  # type: ignore[attr-defined]
					_update_labels._labels_supported = bool(  # type: ignore[attr-defined]
						hasattr(vis, "add_3d_label") and hasattr(vis, "clear_3d_labels")
					)
					_update_labels._warned = False  # type: ignore[attr-defined]
					if not _update_labels._labels_supported and not _update_labels._warned:  # type: ignore[attr-defined]
						print("[viz] 3D labels not supported by this Open3D version; skipping labels.")
						_update_labels._warned = True  # type: ignore[attr-defined]
				if not getattr(_update_labels, "_labels_supported", False):  # type: ignore[attr-defined]
					return
				# Try to clear existing labels
				try:
					vis.clear_3d_labels()
				except Exception:
					if not getattr(_update_labels, "_warned", False):  # type: ignore[attr-defined]
						print("[viz] clear_3d_labels() unavailable; disabling labels.")
						_update_labels._warned = True  # type: ignore[attr-defined]
					_update_labels._labels_supported = False  # type: ignore[attr-defined]
					return
				if P is None or P.size == 0:
					return
				Pd = np.asarray(P, dtype=np.float64)
				finite = np.isfinite(Pd).all(axis=1)
				for j in range(min(Pd.shape[0], len(names_kept))):
					if not finite[j]:
						continue
					try:
						vis.add_3d_label(Pd[j], str(names_kept[j]))
					except Exception:
						if not getattr(_update_labels, "_warned", False):  # type: ignore[attr-defined]
							print("[viz] add_3d_label() unavailable; disabling labels.")
							_update_labels._warned = True  # type: ignore[attr-defined]
						_update_labels._labels_supported = False  # type: ignore[attr-defined]
						break

			def set_frame(vis: o3d.visualization.Visualizer) -> None:
				ti = state["t"]
				P = J3D_INIT[ti] if (state["use_init"] and J3D_INIT is not None) else J3D_OPT[ti]  # (J,3)
				if len(keep_inds) != P.shape[0]:
					P = P[keep_inds]
				pc.points = o3d.utility.Vector3dVector(P.astype(np.float64))
				c = colors_for_frame(outlier[ti])
				pc.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
				if ls is not None:
					ls.points = o3d.utility.Vector3dVector(P.astype(np.float64))
				vis.update_geometry(pc)
				if ls is not None:
					vis.update_geometry(ls)
				_update_labels(vis, P)

			def _cb_space(vis):
				state["playing"] = not state["playing"]
				return False

			def _cb_left(vis):
				state["t"] = (state["t"] - 1) % Ts
				set_frame(vis)
				return False

			def _cb_right(vis):
				state["t"] = (state["t"] + 1) % Ts
				set_frame(vis)
				return False

			def _cb_i(vis):
				state["use_init"] = not state["use_init"]
				set_frame(vis)
				return False

			def _cb_quit(vis):
				vis.close()
				return True

			vis = o3d.visualization.VisualizerWithKeyCallback()
			vis.create_window(window_name="StageB 3D Viewer", width=1280, height=720)
			# Render options
			opt = vis.get_render_option()
			opt.point_size = 8.0
			opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

			# Axis scaled to data
			def data_scale(P: np.ndarray) -> float:
				P = np.asarray(P, dtype=np.float64)
				if P.size == 0:
					return 1.0
				good = np.isfinite(P).all(axis=1)
				if not good.any():
					return 1.0
				C = P[good].mean(axis=0)
				rad = np.linalg.norm(P[good] - C, axis=1).max()
				if not np.isfinite(rad) or rad < 1e-6:
					return 1.0
				return float(rad)

			P_first = np.asarray(P0, dtype=np.float64)
			P_first = P_first[np.isfinite(P_first).all(axis=1)] if P_first.size > 0 else P_first
			s = data_scale(P_first) if P_first.size > 0 else 1.0
			ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25 * s)
			vis.add_geometry(pc)
			if ls is not None:
				vis.add_geometry(ls)
			vis.add_geometry(ax)

			vis.register_key_callback(ord(" "), _cb_space)
			vis.register_key_callback(ord("I"), _cb_i)
			vis.register_key_callback(262, _cb_right)  # GLFW_KEY_RIGHT
			vis.register_key_callback(263, _cb_left)   # GLFW_KEY_LEFT
			vis.register_key_callback(81, _cb_quit)    # Q
			vis.register_key_callback(256, _cb_quit)   # ESC

			# Initial frame
			set_frame(vis)
			vis.reset_view_point(True)

			# Main loop
			last = time.time()
			target_dt = 1.0 / max(1, args.fps)
			while True:
				if not vis.poll_events():
					break
				if state["playing"]:
					now = time.time()
					if now - last >= target_dt:
						state["t"] = (state["t"] + 1) % Ts
						set_frame(vis)
						last = now
				vis.update_renderer()
			vis.destroy_window()
		except Exception as e:
			print("[viz] open3d viewer failed:", e)


if __name__ == "__main__":
	main()


