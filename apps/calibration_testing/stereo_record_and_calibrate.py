"""
Stereo recording + offline calibration + ground-plane fitting pipeline.

- Step 1: Runs the live stereo keyframe recorder to collect keyframes.
- Step 2: Runs the offline stereo calibrator on the recorded keyframes.
- Step 3: Optionally records a short “board on the ground” sequence and fits a ground plane.
- Step 4: Writes a single rig-config JSON that bundles stereo + ground-plane info.

Accepts camera indices, board type, optional Harvard tag size/spacing, and optional
corner order (primarily for the 8x5 grid board).
"""

import os
import sys
import glob
import time
import json
import argparse
import subprocess
import re
import queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
	# Local app imports (for live ground-plane capture)
	from .. import config
	from ..capture import CamReader
	from ..detect import CalibrationAccumulator
	from ..stereo_calib_plot import load_board, solve_pnp_for_view
except Exception:
	# Fallback when run as a script
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from apps import config  # type: ignore
	from apps.capture import CamReader  # type: ignore
	from apps.detect import CalibrationAccumulator  # type: ignore
	from apps.stereo_calib_plot import load_board, solve_pnp_for_view  # type: ignore


def _repo_root() -> str:
	# apps/calibration_testing/ -> apps/ -> repo root
	return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _data_dir() -> str:
	return os.path.join(_repo_root(), "data")


def _latest_dir_with_prefix(base_dir: str, prefix: str) -> str | None:
	if not os.path.isdir(base_dir):
		return None
	candidates = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith(prefix)]
	candidates = [p for p in candidates if os.path.isdir(p)]
	if not candidates:
		return None
	candidates.sort(key=lambda p: os.path.getmtime(p))
	return candidates[-1]


def _latest_file_with_glob(pattern: str) -> str | None:
	paths = glob.glob(pattern)
	if not paths:
		return None
	paths.sort(key=lambda p: os.path.getmtime(p))
	return paths[-1]


def _build_recorder_cmd(args, base_out_dir: str | None) -> list[str]:
	cmd = [
		sys.executable, "-m", "apps.calibration_testing.stereo_keyframe_recorder",
		"--board-source", str(args.board_source),
		"--cam0", str(args.cam0),
		"--cam1", str(args.cam1),
		"--target-keyframes", str(args.target_keyframes),
		"--accept-period", str(args.accept_period),
	]
	if base_out_dir:
		cmd += ["--out-dir", base_out_dir]
	if args.april_pickle:
		cmd += ["--april-pickle", args.april_pickle]
	# Harvard board: tag size (and spacing) may be needed
	if args.board_source == "harvard":
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
	# Grid board: optional corner order override
	if args.board_source == "grid8x5" and args.corner_order:
		cmd += ["--corner-order", args.corner_order]
	return cmd


def _build_calibrator_cmd(args, keyframes_dir: str) -> list[str]:
	cmd = [
		sys.executable, "-m", "apps.calibration_testing.stereo_cam_calibrator_offline",
		"--keyframes-dir", keyframes_dir,
		"--board-source", str(args.board_source),
	]
	if args.april_pickle:
		cmd += ["--april-pickle", args.april_pickle]
	if args.board_source == "harvard":
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
	if args.board_source == "grid8x5" and args.corner_order:
		cmd += ["--corner-order", args.corner_order]
	if args.save_all_diag:
		cmd += ["--save-all-diag"]
	return cmd


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
	"""Fit a plane n·X + d = 0 to 3D points using SVD."""
	pts = np.asarray(points, dtype=np.float64)
	if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
		raise ValueError("Need at least 3x3 points to fit plane")
	centroid = pts.mean(axis=0)
	U, S, Vt = np.linalg.svd(pts - centroid)
	n = Vt[-1, :]
	n_norm = np.linalg.norm(n)
	if n_norm <= 0:
		raise ValueError("Degenerate plane normal")
	n /= n_norm
	d = -float(np.dot(n, centroid))
	return n, d


def _ground_plane_from_live_capture(
	args,
	calib_meta: Dict[str, Any],
	base_out: str,
	target_views: int = 20,
	period_s: float = 1.0,
) -> Optional[Dict[str, Any]]:
	"""
	Use calibrated cam0 intrinsics to capture a short sequence with the board on the ground,
	estimate the ground plane in cam0 frame, and return a dict suitable for JSON.
	"""
	print(f"[GROUND] Starting live ground-plane capture: target_views={target_views}, period={period_s:.1f}s")

	# Open cameras
	print(f"[GROUND] Opening cameras… cam0={args.cam0} cam1={args.cam1}")
	cams = [CamReader(int(args.cam0)), CamReader(int(args.cam1))]
	try:
		ts0, frame0 = cams[0].latest()
	except queue.Empty:
		print("[GROUND][ERR] Could not read from cam0")
		for c in cams:
			c.release()
		return None

	H, W = frame0.shape[:2]
	image_size = (W, H)

	# For floor tags: use individual tags (no board layout)
	# Use floor tag size if provided, otherwise fall back to regular tag size
	floor_tag_size = args.floor_tag_size_m if args.floor_tag_size_m is not None else args.harvard_tag_size_m
	if floor_tag_size is None:
		print("[GROUND][ERR] Floor tag size required (--floor-tag-size-m or --harvard-tag-size-m)")
		for c in cams:
			c.release()
		return None

	# Create a dummy board for the accumulator (needed for detection backend)
	# But we'll create object points dynamically for each detected tag
	board = load_board(
		board_source=args.board_source,
		april_pickle=args.april_pickle,
		harvard_tag_size_m=args.harvard_tag_size_m,
		harvard_tag_spacing_m=args.harvard_tag_spacing_m,
	)

	# Corner-order behavior: mirror recorder defaults
	corner_order_override = None
	disable_autoreorder = False
	if args.corner_order:
		try:
			parts = [int(x.strip()) for x in str(args.corner_order).split(",")]
			if len(parts) == 4 and sorted(parts) == [0, 1, 2, 3]:
				corner_order_override = parts
				disable_autoreorder = True
				print(f"[GROUND][APRIL] Using manual corner order: {corner_order_override}")
			else:
				print(f"[GROUND][WARN] Ignoring invalid --corner-order '{args.corner_order}'")
		except Exception as e:
			print(f"[GROUND][WARN] Failed to parse --corner-order: {e}")
	else:
		# For Harvard, mirror mono default
		if str(args.board_source).lower().strip() == "harvard":
			corner_order_override = [3, 0, 1, 2]
			disable_autoreorder = True
			print("[GROUND][APRIL] Using default per-tag corner order 3,0,1,2 for Harvard board")

	acc = CalibrationAccumulator(
		board,
		image_size,
		corner_order_override=corner_order_override,
		disable_corner_autoreorder=disable_autoreorder,
	)
	print("[GROUND][APRIL] Backend:", acc.get_backend_name())

	# Create object points for individual tags based on floor tag size
	# Tag corners in local frame (centered at origin, Z=0): standard order
	tag_half = float(floor_tag_size) * 0.5
	tag_obj_points_base = np.array([
		[-tag_half, -tag_half, 0.0],  # corner 0
		[ tag_half, -tag_half, 0.0],  # corner 1
		[ tag_half,  tag_half, 0.0],  # corner 2
		[-tag_half,  tag_half, 0.0],  # corner 3
	], dtype=np.float32)
	
	# Apply corner order override if specified
	if corner_order_override is not None:
		tag_obj_points_base = tag_obj_points_base[corner_order_override, :]
	
	print(f"[GROUND] Using individual floor tags with size {floor_tag_size:.4f}m")

	# Intrinsics from offline calibration
	try:
		K0 = np.asarray(calib_meta["K0"], dtype=np.float64)
		D0 = np.asarray(calib_meta["D0"], dtype=np.float64)
	except Exception as e:
		print(f"[GROUND][ERR] Failed to read K0/D0 from calibration JSON: {e}")
		for c in cams:
			c.release()
		return None

	all_pts_cam: List[np.ndarray] = []
	normals: List[np.ndarray] = []

	# Diagnostics directory
	stamp = time.strftime("%Y%m%d_%H%M%S")
	diag_dir = os.path.join(base_out, f"ground_plane_diag_{stamp}")
	os.makedirs(diag_dir, exist_ok=True)

	last_capture = 0.0
	collected = 0

	try:
		while collected < target_views:
			try:
				ts0, frame0 = cams[0].latest()
			except queue.Empty:
				time.sleep(0.01)
				continue

			now = time.perf_counter()
			if (now - last_capture) < float(period_s):
				time.sleep(0.01)
				continue
			last_capture = now

			gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
			corners0, ids0 = acc.detect(gray0)
			n_tags = 0 if ids0 is None else len(ids0)
			if ids0 is None or not corners0 or n_tags < 4:
				print(f"[GROUND] View {collected+1}: not enough tags (have {n_tags}, need 4); skipping")
				continue

			# Build per-tag object/image corner lists
			# For floor tags: create object points dynamically for each detected tag
			obj_list: List[np.ndarray] = []
			img_list: List[np.ndarray] = []
			for c, iv in zip(corners0, ids0):
				tid = int(iv[0])
				# Use floor tag object points (same for all tags since they're individual)
				obj_list.append(tag_obj_points_base.copy())
				img_list.append(c.reshape(4, 2))
			if not obj_list:
				print(f"[GROUND] View {collected+1}: no mappable tags; skipping")
				continue

			rvec, tvec = solve_pnp_for_view(K0, D0, obj_list, img_list)
			if rvec is None or tvec is None:
				print(f"[GROUND] View {collected+1}: solvePnP failed; skipping")
				continue

			R, _ = cv2.Rodrigues(rvec)
			t = tvec.reshape(3,)

			# Transform all board points to camera frame
			obj_cat = np.concatenate(obj_list, axis=0)  # (N,3)
			pts_cam = (R @ obj_cat.T + t.reshape(3, 1)).T  # (N,3)
			all_pts_cam.append(pts_cam)

			# Per-view normal (board Z axis in camera frame)
			n_view = R[:, 2].astype(np.float64)
			n_view_norm = np.linalg.norm(n_view)
			if n_view_norm > 0:
				n_view /= n_view_norm
				normals.append(n_view)

			# Simple diagnostic image with projected points
			try:
				vis = frame0.copy()
				for c in corners0:
					cv2.polylines(vis, [c.reshape(4, 2).astype(np.int32)], True, (0, 255, 255), 2)
				out_png = os.path.join(diag_dir, f"ground_view_{collected:03d}.png")
				cv2.imwrite(out_png, vis)
			except Exception:
				pass

			collected += 1
			print(f"[GROUND] Collected {collected}/{target_views} valid ground views (tags={n_tags})")
	finally:
		for c in cams:
			c.release()

	if not all_pts_cam:
		print("[GROUND][ERR] No valid ground-plane views collected; skipping plane fit.")
		return None

	pts_all = np.concatenate(all_pts_cam, axis=0)
	try:
		n, d = _fit_plane_svd(pts_all)
	except Exception as e:
		print(f"[GROUND][ERR] Plane fit failed: {e}")
		return None

	# Flip normal, if needed, so that it points "up" (positive Y direction).
	# Check: if normal · [0,1,0] < 0, flip normal and d.
	world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
	dot_product = np.dot(n, world_up)
	if dot_product < 0:
		n = -n
		d = -d

	gp: Dict[str, Any] = {
		"normal_cam": [float(v) for v in n],
		"d": float(d),  # plane: n·X + d = 0 in cam0 frame
		"num_points": int(pts_all.shape[0]),
		"num_views": int(len(all_pts_cam)),
	}
	print(f"[GROUND] Fitted plane: n={gp['normal_cam']}  d={gp['d']:.6f}")
	return gp


def _infer_stamp_from_calib_path(calib_json: str) -> str:
	m = re.search(r"stereo_offline_calibration_(\d{8}_\d{6})\.json$", os.path.basename(calib_json))
	if m:
		return m.group(1)
	return time.strftime("%Y%m%d_%H%M%S")


def _build_rig_config(
	calib_meta: Dict[str, Any],
	ground_plane: Optional[Dict[str, Any]],
	rig_id: str,
) -> Dict[str, Any]:
	"""Combine stereo calibration JSON + optional ground-plane into a rig-config dict."""
	def _as_nd(x: Any) -> np.ndarray:
		return np.asarray(x, dtype=np.float64)

	K0 = _as_nd(calib_meta.get("K0"))
	D0 = _as_nd(calib_meta.get("D0")).reshape(-1)
	K1 = _as_nd(calib_meta.get("K1"))
	D1 = _as_nd(calib_meta.get("D1")).reshape(-1)
	R = _as_nd(calib_meta.get("R"))
	T = _as_nd(calib_meta.get("T")).reshape(3,)

	rig: Dict[str, Any] = {
		"version": 1,
		"rig_id": rig_id,
		"image_size": calib_meta.get("image_size"),
		"stereo_calib": {
			"camera_0": {
				"K": K0.tolist(),
				"dist_coeffs": D0.tolist(),
				"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
				"t": [0.0, 0.0, 0.0],
			},
			"camera_1": {
				"K": K1.tolist(),
				"dist_coeffs": D1.tolist(),
				"R": R.tolist(),
				"t": T.tolist(),
			},
		},
	}

	# Ground plane payload in cam0 frame
	gp: Dict[str, Any] = {}
	if ground_plane:
		gp.update(ground_plane)
	# Defaults that can be overridden later by viewers / downstream tools
	gp.setdefault("offset_to_floor_m", 0.0)
	gp.setdefault("world_up", [0.0, 1.0, 0.0])
	gp.setdefault("world_origin", [0.0, 0.0, 0.0])
	rig["ground_plane"] = gp

	return rig


def main():
	parser = argparse.ArgumentParser(description="Stereo pipeline: record keyframes, offline calibrate, then estimate ground plane -> rig config JSON.")
	# Cameras
	parser.add_argument("--cam0", type=int, default=0, help="Camera index for left/first camera.")
	parser.add_argument("--cam1", type=int, default=1, help="Camera index for right/second camera.")
	# Board
	parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], default="harvard",
	                    help="Calibration board type.")
	parser.add_argument("--april-pickle", type=str, default=None,
	                    help="Path to local Harvard AprilBoards.pickle (only used for Harvard board).")
	parser.add_argument("--harvard-tag-size-m", type=float, default=None,
	                    help="Tag side length in meters (Harvard board).")
	parser.add_argument("--harvard-tag-spacing-m", type=float, default=None,
	                    help="Tag spacing in meters (Harvard board, optional; improves scaling from centers data).")
	parser.add_argument("--floor-tag-size-m", type=float, default=None,
	                    help="Tag side length in meters for floor calibration (individual tags, not a board).")
	parser.add_argument("--corner-order", type=str, default=None,
	                    help="Corner order override 'i0,i1,i2,i3' (only applicable to grid8x5).")
	# Recording
	parser.add_argument("--target-keyframes", type=int, default=50, help="Number of keyframes to collect.")
	parser.add_argument("--accept-period", type=float, default=0.5, help="Minimum seconds between accepted keyframes.")
	parser.add_argument("--out-dir", type=str, default=None, help="Base output directory (default: repo_root/data).")
	# Calibrator diagnostics
	parser.add_argument("--save-all-diag", action="store_true", help="Save diagnostic overlays for all kept views.")
	# Rig metadata
	parser.add_argument("--rig-id", type=str, default=None, help="Optional rig identifier for rig-config JSON.")

	args = parser.parse_args()

	# Resolve base data directory
	base_out = os.path.abspath(args.out_dir) if args.out_dir else _data_dir()
	os.makedirs(base_out, exist_ok=True)
	print(f"[PIPE] Using base output directory: {base_out}")

	# Helpful warnings about optional parameters
	if args.board_source == "grid8x5" and args.harvard_tag_size_m is not None:
		print("[PIPE][WARN] --harvard-tag-size-m provided but board-source is grid8x5; ignoring.")
	if args.board_source == "harvard" and args.corner_order:
		print("[PIPE][WARN] --corner-order provided but primarily used for grid8x5; continuing.")

	# 1) Record keyframes
	rec_cmd = _build_recorder_cmd(args, base_out)
	print("[PIPE] Launching stereo keyframe recorder...")
	try:
		subprocess.run(rec_cmd, check=True)
	except subprocess.CalledProcessError as e:
		print(f"[PIPE][ERR] Keyframe recorder failed with code {e.returncode}")
		sys.exit(1)

	# Find the most recent keyframes directory
	keyframes_dir = _latest_dir_with_prefix(base_out, "stereo_keyframes_")
	if not keyframes_dir:
		print("[PIPE][ERR] Could not find a newly created 'stereo_keyframes_*' directory.")
		sys.exit(1)
	print(f"[PIPE] Using keyframes directory: {keyframes_dir}")

	# 2) Run offline calibration
	cal_cmd = _build_calibrator_cmd(args, keyframes_dir)
	print("[PIPE] Launching stereo offline calibrator...")
	try:
		subprocess.run(cal_cmd, check=True)
	except subprocess.CalledProcessError as e:
		print(f"[PIPE][ERR] Calibrator failed with code {e.returncode}")
		sys.exit(1)

	# 3) Locate the most recent calibration JSON
	calib_json = _latest_file_with_glob(os.path.join(_data_dir(), "stereo_offline_calibration_*.json"))
	if not calib_json:
		print("[PIPE][ERR] Calibration JSON not found in data/.")
		sys.exit(2)

	# Echo summary and keep metadata for rig-config
	try:
		with open(calib_json, "r", encoding="utf-8") as f:
			meta = json.load(f)
		image_size = tuple(meta.get("image_size", []))
		board_source = meta.get("board_source", "unknown")
		print(f"[PIPE] Calibration complete. JSON: {calib_json}")
		print(f"[PIPE] image_size={image_size} board_source={board_source}")
	except Exception:
		print(f"[PIPE] Calibration complete. JSON: {calib_json}")
		meta = None

	if meta is None:
		# Nothing more we can do for rig-config
		return

	# 4) Ground-plane capture stage (interactive)
	print()
	print("[PIPE] === Ground-plane estimation ===")
	print("[PIPE] Place the calibration board flat on the ground in front of the rig.")
	print("[PIPE] When ready, this script will capture 20 views from cam0, 1 second apart.")
	try:
		input("[PIPE] Press ENTER in this terminal to start ground-plane capture, or Ctrl+C to skip... ")
	except (EOFError, KeyboardInterrupt):
		print("[PIPE][WARN] Ground-plane capture skipped by user/EOF.")
		ground_plane = None
	else:
		ground_plane = _ground_plane_from_live_capture(args, meta, base_out, target_views=20, period_s=1.0)

	# 5) Build and save rig-config JSON
	stamp = _infer_stamp_from_calib_path(calib_json)
	rig_id = args.rig_id or f"rig_{stamp}"
	rig_cfg = _build_rig_config(meta, ground_plane, rig_id)
	rig_json = os.path.join(base_out, f"rig_config_{stamp}.json")
	with open(rig_json, "w", encoding="utf-8") as f:
		json.dump(rig_cfg, f, indent=2)
	print(f"[PIPE] Rig-config JSON written to: {rig_json}")


if __name__ == "__main__":
	main()


