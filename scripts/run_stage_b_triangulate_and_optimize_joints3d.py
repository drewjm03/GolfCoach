from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import cv2
from tqdm import tqdm
import hashlib

from golfcoach.io.npz_io import load_npz, save_npz_compressed
from golfcoach.io.rig_config import load_rig_config
from golfcoach.pose3d.joints3d_kf_triang_opt import (
	OptimizeConfig,
	build_bone_edges_for_height_5ft10,
	kalman_filter_gating_2d,
	optimize_joints3d_with_priors,
	rmse_reprojection_per_joint,
	triangulate_init,
	undistort_kpts_pixels,
)


def _hash_array(a: np.ndarray) -> str:
	b = np.ascontiguousarray(a).view(np.uint8)
	return hashlib.sha1(b).hexdigest()[:10]


def _raw_to_used(
	u_raw: np.ndarray,
	K: np.ndarray,
	D: np.ndarray,
	src_size: Tuple[int, int],
	dst_size: Tuple[int, int],
	dbg: bool = False,
) -> np.ndarray:
	"""
	Convert raw pixel coords to undistorted pixel coords using cv2.undistortPoints with P=K.
	We scale intrinsics externally if src_size != dst_size, so u_raw is not scaled here.
	"""
	assert u_raw.shape == (2,), f"u_raw expected (2,), got {u_raw.shape}"
	if dbg:
		Ws, Hs = int(src_size[0]), int(src_size[1])
		Wd, Hd = int(dst_size[0]), int(dst_size[1])
		sx = (Wd / Ws) if Ws > 0 else float("nan")
		sy = (Hd / Hs) if Hs > 0 else float("nan")
		print(f"[raw_to_used] src_size={src_size}, dst_size={dst_size}, sx={sx:.6f}, sy={sy:.6f}")
		print("[raw_to_used] OpenCV undistort: cv2.undistortPoints with P=K (pinhole, pixel space)")
		print("[raw_to_used] K fx,cx,fy,cy:", float(K[0,0]), float(K[0,2]), float(K[1,1]), float(K[1,2]))
		print("[raw_to_used] dist (first 5):", np.asarray(D, dtype=np.float64).ravel()[:5])
		print("[raw_to_used] u_after_scale_pre_undistort (same as u_raw):", u_raw)
	u1 = u_raw.reshape(1, 1, 2).astype(np.float64)
	und = cv2.undistortPoints(u1, K.astype(np.float64), D.astype(np.float64), P=K.astype(np.float64))
	u_used = und.reshape(2).astype(np.float64)
	if dbg:
		print("[raw_to_used] u_after_undistort (direct):", u_used)
		print("[raw_to_used] u_final_used:", u_used)
	return u_used


def safe_div(num: np.ndarray, den: float, eps: float = 1e-12) -> np.ndarray:
	den = float(den)
	if abs(den) < eps:
		den = eps if den >= 0.0 else -eps
	return num / den


def _align_by_frame_idx(left: Dict, right: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], Tuple[int, int]]:
	# Extract basic arrays
	tL = np.asarray(left["t"]).astype(np.float64)
	iL = np.asarray(left["frame_idx"]).astype(np.int64)
	kL = np.asarray(left["kpts"]).astype(np.float32)
	cL = np.asarray(left["conf"]).astype(np.float32)
	szL = tuple(np.asarray(left["image_size"]).tolist())
	jL = [str(x) for x in np.asarray(left["joint_names"]).tolist()]

	tR = np.asarray(right["t"]).astype(np.float64)
	iR = np.asarray(right["frame_idx"]).astype(np.int64)
	kR = np.asarray(right["kpts"]).astype(np.float32)
	cR = np.asarray(right["conf"]).astype(np.float32)
	szR = tuple(np.asarray(right["image_size"]).tolist())
	jR = [str(x) for x in np.asarray(right["joint_names"]).tolist()]

	if jL != jR:
		raise ValueError("Left/right joint_names mismatch")
	if szL != szR:
		raise ValueError(f"Left/right image_size mismatch: {szL} vs {szR}")

	# Map by frame_idx
	mapL = {int(fi): ti for ti, fi in enumerate(iL.tolist())}
	mapR = {int(fi): ti for ti, fi in enumerate(iR.tolist())}
	common_idx = sorted(set(mapL.keys()).intersection(set(mapR.keys())))
	if len(common_idx) == 0:
		raise ValueError("No overlapping frame_idx between left and right sequences")

	T = len(common_idx)
	J = kL.shape[1]
	t = np.zeros((T,), dtype=np.float64)
	kptsL = np.zeros((T, J, 2), dtype=np.float32)
	kptsR = np.zeros((T, J, 2), dtype=np.float32)
	confL = np.zeros((T, J), dtype=np.float32)
	confR = np.zeros((T, J), dtype=np.float32)
	idx_arr = np.zeros((T,), dtype=np.int64)
	for ti, fi in enumerate(common_idx):
		li = mapL[fi]
		ri = mapR[fi]
		# Prefer left timestamps; optionally average
		t[ti] = float(tL[li])  # or (tL[li] + tR[ri]) * 0.5
		kptsL[ti] = kL[li]
		kptsR[ti] = kR[ri]
		confL[ti] = cL[li]
		confR[ti] = cR[ri]
		idx_arr[ti] = fi

	return t.astype(np.float64), dict(kpts=kptsL, conf=confL), dict(kpts=kptsR, conf=confR), jL, szL


def debug_single_frame_triangulate_and_reproj(
	ti: int,
	kptsL_ud: np.ndarray, kptsR_ud: np.ndarray,       # (T,J,2) undistorted
	kptsL_raw: np.ndarray | None, kptsR_raw: np.ndarray | None,  # (T,J,2) original (optional)
	confL: np.ndarray, confR: np.ndarray,             # (T,J)
	validL_kf: np.ndarray, validR_kf: np.ndarray,     # (T,J)
	K0: np.ndarray, D0: np.ndarray, K1: np.ndarray, D1: np.ndarray,
	Rst: np.ndarray, Tst: np.ndarray,
	joint_names: Sequence[str],
	image_size: Tuple[int, int],
	conf_min: float,
	use_kf_mask: bool,
	print_all: bool,
) -> None:
	T, J, _ = kptsL_ud.shape
	assert 0 <= ti < T, f"ti out of range: {ti} (T={T})"

	uL = kptsL_ud[ti].astype(np.float64)  # (J,2)
	uR = kptsR_ud[ti].astype(np.float64)
	uL_raw = kptsL_raw[ti].astype(np.float64) if kptsL_raw is not None else None
	uR_raw = kptsR_raw[ti].astype(np.float64) if kptsR_raw is not None else None

	# conf-only masks (recommended)
	finiteL = np.isfinite(uL).all(axis=1)
	finiteR = np.isfinite(uR).all(axis=1)
	mL_conf = (confL[ti] >= conf_min) & finiteL
	mR_conf = (confR[ti] >= conf_min) & finiteR

	# KF masks
	mL_kf = validL_kf[ti].astype(bool) & finiteL
	mR_kf = validR_kf[ti].astype(bool) & finiteR

	mL = mL_kf if use_kf_mask else mL_conf
	mR = mR_kf if use_kf_mask else mR_conf

	both = mL & mR
	print("\n================ SINGLE-FRAME TRIANGULATION DEBUG ================")
	print(f"ti={ti}, use_kf_mask={use_kf_mask}")
	print(f"conf-only valid L/R: {int(mL_conf.sum())}/{int(mR_conf.sum())}  both={int((mL_conf & mR_conf).sum())}")
	print(f"KF-gated  valid L/R: {int(mL_kf.sum())}/{int(mR_kf.sum())}  both={int((mL_kf & mR_kf).sum())}")
	print(f"USING mask L/R:      {int(mL.sum())}/{int(mR.sum())}        both={int(both.sum())}")
	print("image_size:", image_size)
	print("K0 fx,cx,fy,cy:", float(K0[0,0]), float(K0[0,2]), float(K0[1,1]), float(K0[1,2]))
	print("K1 fx,cx,fy,cy:", float(K1[0,0]), float(K1[0,2]), float(K1[1,1]), float(K1[1,2]))
	print("||Tst||:", float(np.linalg.norm(Tst.reshape(3))))

	# Print per-joint 2D keypoints (raw and undistorted)
	print("\n-- 2D keypoints (ti=%d) --" % ti)
	for j in range(J):
		name = str(joint_names[j])
		if uL_raw is not None and uR_raw is not None:
			print(f"{j:02d} {name:>16s}  L_raw={uL_raw[j]}  R_raw={uR_raw[j]}  L_used={uL[j]}  R_used={uR[j]}")
		else:
			print(f"{j:02d} {name:>16s}  L_used={uL[j]}  R_used={uR[j]}")

	# Build camera matrices
	P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
	P1 = K1 @ np.hstack([Rst, Tst.reshape(3, 1)])

	def triangulate_dlt(uL2: np.ndarray, uR2: np.ndarray) -> np.ndarray:
		xL = np.array([uL2[0], uL2[1], 1.0], dtype=np.float64)
		xR = np.array([uR2[0], uR2[1], 1.0], dtype=np.float64)
		A = np.stack([
			xL[0] * P0[2] - P0[0],
			xL[1] * P0[2] - P0[1],
			xR[0] * P1[2] - P1[0],
			xR[1] * P1[2] - P1[1],
		], axis=0)
		_, _, Vt = np.linalg.svd(A)
		Xh = Vt[-1]
		w = float(Xh[3])
		den = w if abs(w) > 1e-12 else (1e-12 if w >= 0 else -1e-12)
		return (Xh[:3] / den).astype(np.float64)

	def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
		Xh = np.append(X, 1.0)
		x = P @ Xh
		return safe_div(x[:2], x[2], eps=1e-12).astype(np.float64)

	# Triangulate + reproj errors
	z_min = 0.2
	reproj_thresh = 20.0
	tri_ok = 0
	reject_z = 0
	reject_reproj = 0

	errsL: List[float] = []
	errsR: List[float] = []

	for j in range(J):
		name = str(joint_names[j])
		if not both[j]:
			if print_all:
				print(f"{j:02d} {name:>16s}  SKIP  confL/R={confL[ti,j]:.3f}/{confR[ti,j]:.3f}  "
					f"mL={bool(mL[j])} mR={bool(mR[j])}  uL={uL[j]} uR={uR[j]}")
			continue

		X = triangulate_dlt(uL[j], uR[j])
		XR = (Rst @ X + Tst.reshape(3))
		zL = float(X[2]); zR = float(XR[2])
		if not (np.isfinite(X).all() and np.isfinite(XR).all() and zL > z_min and zR > z_min):
			reject_z += 1
			if print_all:
				print(f"{j:02d} {name:>16s}  REJECT_Z  zL/zR={zL:.3f}/{zR:.3f}  X={X}")
			continue

		uL_hat = project(P0, X)
		uR_hat = project(P1, X)
		eL = float(np.linalg.norm(uL_hat - uL[j]))
		eR = float(np.linalg.norm(uR_hat - uR[j]))

		if eL > reproj_thresh or eR > reproj_thresh:
			reject_reproj += 1
			if print_all:
				print(f"{j:02d} {name:>16s}  REJECT_REPROJ  eL/eR={eL:.2f}/{eR:.2f}  "
					f"uL={uL[j]} uLhat={uL_hat}  uR={uR[j]} uRhat={uR_hat}")
			continue

		tri_ok += 1
		errsL.append(eL)
		errsR.append(eR)
		print(f"{j:02d} {name:>16s}  OK  zL/zR={zL:.3f}/{zR:.3f}  eL/eR={eL:.2f}/{eR:.2f}")

	def stats(a: List[float]) -> str:
		if len(a) == 0:
			return "none"
		arr = np.array(a, dtype=np.float64)
		return f"n={len(arr)} mean={arr.mean():.2f} med={np.median(arr):.2f} p95={np.percentile(arr,95):.2f}"

	print("---- summary ----")
	print(f"both2d={int(both.sum())} tri_ok={tri_ok} reject_z={reject_z} reject_reproj={reject_reproj}")
	print("reproj L:", stats(errsL))
	print("reproj R:", stats(errsR))
	print("===============================================================\n")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--left_npz", type=str, required=True)
	ap.add_argument("--right_npz", type=str, required=True)
	ap.add_argument("--rig_json", type=str, required=True)
	ap.add_argument("--out_npz", type=str, default="")
	ap.add_argument("--conf_min", type=float, default=0.05)
	ap.add_argument("--sigma_min_px", type=float, default=10.0)
	ap.add_argument("--sigma_max_px", type=float, default=80.0)
	ap.add_argument("--q_accel", type=float, default=1e6)
	ap.add_argument("--win", type=int, default=30, help="Sliding window size (frames)")
	ap.add_argument("--stride", type=int, default=10, help="Sliding window stride (frames)")
	ap.add_argument("--lambda_bone", type=float, default=300.0, help="Weight for bone-length prior")
	ap.add_argument("--lambda_acc", type=float, default=1e-2, help="Weight for constant-acceleration prior")
	ap.add_argument("--debug_frame", type=int, default=-1, help="Aligned time index ti to debug (e.g. 79). -1 disables.")
	ap.add_argument("--debug_use_kf_mask", action="store_true", help="Use KF 'valid' masks instead of conf-only for debug frame.")
	ap.add_argument("--debug_print_all_joints", action="store_true", help="Print per-joint lines even if invalid.")
	args = ap.parse_args()

	left = load_npz(args.left_npz)
	right = load_npz(args.right_npz)

	# Alignment verification (pre-alignment diagnostics)
	try:
		import numpy as _np
		_Lraw = _np.load(str(args.left_npz), allow_pickle=True)
		_Rraw = _np.load(str(args.right_npz), allow_pickle=True)
		T_left = len(_Lraw["t"])
		T_right = len(_Rraw["t"])
		print("T_left, T_right:", T_left, T_right)
		minT = min(len(_Lraw["frame_idx"]), len(_Rraw["frame_idx"]))
		f_eq = bool(_np.array_equal(_Lraw["frame_idx"], _Rraw["frame_idx"]))
		max_fdiff = int(_np.max(_np.abs(_Lraw["frame_idx"][:minT] - _Rraw["frame_idx"][:minT]))) if minT > 0 else 0
		minTt = min(len(_Lraw["t"]), len(_Rraw["t"]))
		max_tdiff = float(_np.max(_np.abs(_Lraw["t"][:minTt] - _Rraw["t"][:minTt]))) if minTt > 0 else 0.0
		print("frame_idx equal:", f_eq)
		print("max |frame_idx diff|:", max_fdiff)
		print("max |t diff|:", max_tdiff)
	except Exception as e:
		print("[WARN] Alignment verification skipped:", e)

	# Align by frame index
	t, L, R, joint_names, image_size = _align_by_frame_idx(left, right)
	frame_idx = np.arange(len(t), dtype=np.int32)  # informational; aligned implicit order
	# Print first ~10 joint names once
	try:
		print("first 10 joint names:", [str(n) for n in joint_names[:10]])
	except Exception:
		pass

	# Load rig config (K, D, R, T)
	img_sz_rig, K0, D0, K1, D1, Rst, Tst = load_rig_config(args.rig_json)
	if tuple(img_sz_rig) != tuple(image_size):
		print(f"[WARN] Rig image_size {img_sz_rig} differs from pose2d {image_size}")
	print("npz left image_size:", left["image_size"])
	print("npz right image_size:", right["image_size"])
	print("rig image_size:", img_sz_rig)
	# If different, scale intrinsics to npz resolution
	Wc, Hc = int(img_sz_rig[0]), int(img_sz_rig[1])
	Wn, Hn = int(image_size[0]), int(image_size[1])
	if (Wc, Hc) != (Wn, Hn):
		sx = float(Wn) / float(Wc) if Wc > 0 else 1.0
		sy = float(Hn) / float(Hc) if Hc > 0 else 1.0
		print(f"[INFO] Scaling intrinsics by (sx, sy)=({sx:.6f}, {sy:.6f}) to match npz resolution")
		K0 = K0.copy().astype(np.float64)
		K1 = K1.copy().astype(np.float64)
		K0[0, 0] *= sx  # fx
		K0[0, 2] *= sx  # cx
		K0[1, 1] *= sy  # fy
		K0[1, 2] *= sy  # cy
		K1[0, 0] *= sx
		K1[0, 2] *= sx
		K1[1, 1] *= sy
		K1[1, 2] *= sy

	# Undistort both cameras in pixel coordinate system
	kptsL_ud = undistort_kpts_pixels(L["kpts"], K0, D0)
	kptsR_ud = undistort_kpts_pixels(R["kpts"], K1, D1)

	# Single-frame truth test (optional diag)
	try:
		confL = L["conf"]
		confR = R["conf"]
		score = (confL * confR).sum(axis=1)
		t_best = int(np.argmax(score))
		joint_names_list = list(joint_names)
		j_try = joint_names_list.index("thorax") if "thorax" in joint_names_list else 0
		print("truth test: best frame:", t_best, "joint index/name:", j_try, joint_names_list[j_try])
		# Raw vs undistorted/used for that frame/joint
		uL_raw_truth = L["kpts"][t_best, j_try].astype(np.float64)
		uR_raw_truth = R["kpts"][t_best, j_try].astype(np.float64)
		# Derive u_used via the same function
		uL_used_truth = _raw_to_used(uL_raw_truth, K0, D0, img_sz_rig, image_size, dbg=True)
		uR_used_truth = _raw_to_used(uR_raw_truth, K1, D1, img_sz_rig, image_size, dbg=True)
		print("uL_raw/uR_raw:", uL_raw_truth, uR_raw_truth, "conf:", confL[t_best, j_try], confR[t_best, j_try])
		print("uL_used/uR_used:", uL_used_truth, uR_used_truth)
		# Intrinsics/dist used
		print("truth path K0 fx,cx,fy,cy:", float(K0[0,0]), float(K0[0,2]), float(K0[1,1]), float(K0[1,2]), "dist0[:5]:", np.asarray(D0).ravel()[:5], "img_sz:", image_size)
		print("truth path K1 fx,cx,fy,cy:", float(K1[0,0]), float(K1[0,2]), float(K1[1,1]), float(K1[1,2]), "dist1[:5]:", np.asarray(D1).ravel()[:5], "img_sz:", image_size)
		# DLT triangulation and reproj (numpy only)
		def triangulate_dlt(uL: np.ndarray, uR: np.ndarray, K_L: np.ndarray, K_R: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
			P0 = K_L @ np.hstack([np.eye(3), np.zeros((3, 1))])
			P1 = K_R @ np.hstack([R, t.reshape(3, 1)])
			xL = np.array([uL[0], uL[1], 1.0], dtype=np.float64)
			xR = np.array([uR[0], uR[1], 1.0], dtype=np.float64)
			A = np.stack([
				xL[0] * P0[2] - P0[0],
				xL[1] * P0[2] - P0[1],
				xR[0] * P1[2] - P1[0],
				xR[1] * P1[2] - P1[1],
			], axis=0)
			_, _, Vt = np.linalg.svd(A)
			Xh = Vt[-1]
			# Enforce positive w
			if float(Xh[3]) < 0.0:
				Xh = -Xh
			w = float(Xh[3])
			den = w if abs(w) > 1e-12 else (1e-12 if w >= 0 else -1e-12)
			X = (Xh[:3] / den).astype(np.float64)
			return X
		def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
			Xh = np.append(X, 1.0)
			x = P @ Xh
			return safe_div(x[:2], x[2], eps=1e-12).astype(np.float64)
		def reproj_err(u: np.ndarray, uhat: np.ndarray) -> float:
			return float(np.linalg.norm(u - uhat))
		uL_px = kptsL_ud[t_best, j_try].astype(np.float64)
		uR_px = kptsR_ud[t_best, j_try].astype(np.float64)
		X = triangulate_dlt(uL_px, uR_px, K0.astype(np.float64), K1.astype(np.float64), Rst.astype(np.float64), Tst.astype(np.float64))
		XR = (Rst @ X + Tst.reshape(3)).astype(np.float64)
		P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
		P1 = K1 @ np.hstack([Rst.astype(np.float64), Tst.reshape(3, 1).astype(np.float64)])
		uL_hat = project(P0, X)
		uR_hat = project(P1, X)
		eL = reproj_err(uL_px, uL_hat)
		eR = reproj_err(uR_px, uR_hat)
		print("truth X_L:", X, "zL/zR:", X[2], XR[2], "reproj px L/R:", f"{eL:.2f}", f"{eR:.2f}")
		# R,t direction sanity using known-good truth X
		P1R = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
		X_R_a = (Rst @ X + Tst.reshape(3)).astype(np.float64)
		X_R_b = (Rst.T @ (X - Tst.reshape(3))).astype(np.float64)
		uR_hat_a = project(P1R, X_R_a)
		uR_hat_b = project(P1R, X_R_b)
		eR_a = reproj_err(uR_px, uR_hat_a)
		eR_b = reproj_err(uR_px, uR_hat_b)
		print("R,t direction test (right view): eR_a=R@X+t:", f"{eR_a:.2f}", "eR_b=R.T@(X-t):", f"{eR_b:.2f}")
	except Exception as e:
		print("[WARN] Truth test skipped:", e)

	# KF + gating on each view
	kfL = kalman_filter_gating_2d(
		t=t,
		kpts=kptsL_ud,
		conf=L["conf"],
		conf_min=args.conf_min,
		sigma_min_px=args.sigma_min_px,
		sigma_max_px=args.sigma_max_px,
		q_accel=args.q_accel,
		gate_thresh=100.0,
	)
	kfR = kalman_filter_gating_2d(
		t=t,
		kpts=kptsR_ud,
		conf=R["conf"],
		conf_min=args.conf_min,
		sigma_min_px=args.sigma_min_px,
		sigma_max_px=args.sigma_max_px,
		q_accel=args.q_accel,
		gate_thresh=100.0,
	)
	kL_f = kfL["kpts_filt"]
	kR_f = kfR["kpts_filt"]
	validL = kfL["valid"]
	validR = kfR["valid"]

	# KF diagnostics at ti=79 (if in range)
	try:
		ti_dbg = 79
		if 0 <= ti_dbg < len(t):
			dt_dbg = float(t[ti_dbg] - t[ti_dbg - 1]) if ti_dbg > 0 else float("nan")
			print("KF dbg ti:", ti_dbg)
			print("dt:", dt_dbg)
			print("KF valid count L/R:", int(validL[ti_dbg].sum()), int(validR[ti_dbg].sum()))
			print("KF rejected count L/R:", int(kfL["was_rejected"][ti_dbg].sum()), int(kfR["was_rejected"][ti_dbg].sum()))
			d2L = np.asarray(kfL["d2"][ti_dbg], dtype=np.float64)
			d2R = np.asarray(kfR["d2"][ti_dbg], dtype=np.float64)
			print("d2 stats L:", float(np.nanmin(d2L)), float(np.nanmedian(d2L)), float(np.nanmax(d2L)))
			print("d2 stats R:", float(np.nanmin(d2R)), float(np.nanmedian(d2R)), float(np.nanmax(d2R)))
	except Exception as _e:
		print("[WARN] KF debug dump failed:", _e)

	# Single frame debug (conf-only vs KF masks), then exit
	if args.debug_frame >= 0:
		debug_single_frame_triangulate_and_reproj(
			ti=int(args.debug_frame),
			kptsL_ud=kptsL_ud,
			kptsR_ud=kptsR_ud,
			kptsL_raw=L["kpts"],
			kptsR_raw=R["kpts"],
			confL=L["conf"],
			confR=R["conf"],
			validL_kf=validL,
			validR_kf=validR,
			K0=K0, D0=D0, K1=K1, D1=D1,
			Rst=Rst, Tst=Tst,
			joint_names=joint_names,
			image_size=image_size,
			conf_min=float(args.conf_min),
			use_kf_mask=bool(args.debug_use_kf_mask),
			print_all=bool(args.debug_print_all_joints),
		)
		return

	# Triangulation init
	# Force bulk debug for specific frame/joint if desired
	DBG_F = 81
	DBG_JNAME = "thorax"
	X_init, valid3d_init, tri_counts = triangulate_init(
		kL_f, kR_f, validL, validR, K0, K1, Rst, Tst,
		joint_names=joint_names,
		dbg_ti=DBG_F,
		dbg_joint_name=DBG_JNAME,
	)
	print("tri counts:", tri_counts)

	# Sliding window optimization
	T = len(t)
	win = int(max(1, args.win))
	stride = int(max(1, args.stride))
	X_accum = np.zeros_like(X_init, dtype=np.float64)
	N_accum = np.zeros((T, X_init.shape[1], 1), dtype=np.float64)
	scale_weighted_sum = 0.0
	scale_weight = 0.0
	last_losses = {}
	print(f"[StageB] Sliding window optimization: win={win}, stride={stride}, T={T}")
	for s in range(0, T, stride):
		end = min(T, s + win)
		if end - s < 2:
			break
		progress_tag = f"win {s}:{end}"
		# Quick debug prints before optimization
		try:
			t0 = float(t[s])
			t1 = float(t[end - 1])
		except Exception:
			t0, t1 = float(s), float(end - 1)
		maskL_w = validL[s:end]
		maskR_w = validR[s:end]
		xw = X_init[s:end]
		finite_frac = float(np.isfinite(xw).mean()) if xw.size > 0 else float("nan")
		zw = xw[..., 2]
		if np.isfinite(zw).any():
			zmin = float(np.nanmin(zw))
			zmed = float(np.nanmedian(zw))
		else:
			zmin = float("nan")
			zmed = float("nan")
		print(
			f"window {t0:.3f} {t1:.3f} "
			f"valid2d_L {int(maskL_w.sum())} "
			f"valid2d_R {int(maskR_w.sum())}"
		)
		print(
			f"X_init finite% {finite_frac:.3f} "
			f"zL min/med {zmin:.3f} {zmed:.3f}"
		)
		# Minimal diagnostic: compare optimizer projection path vs DLT truth on a fixed frame/joint
		if s == 0:
			try:
				DBG_F = 81
				DBG_JNAME = "thorax"
				joint_names_list = list(joint_names)
				dbg_j = joint_names_list.index(DBG_JNAME) if DBG_JNAME in joint_names_list else 0
				print(f"[{progress_tag}] OPT PATH CHECK joint index/name:", dbg_j, joint_names_list[dbg_j])
				# Raw vs used for optimizer inputs
				uL_raw_opt = L["kpts"][DBG_F, dbg_j].astype(np.float64)
				uR_raw_opt = R["kpts"][DBG_F, dbg_j].astype(np.float64)
				uL_px = _raw_to_used(uL_raw_opt, K0, D0, img_sz_rig, image_size, dbg=True)
				uR_px = _raw_to_used(uR_raw_opt, K1, D1, img_sz_rig, image_size, dbg=True)
				print(f"[{progress_tag}] uL_raw/uR_raw:", uL_raw_opt, uR_raw_opt)
				print(f"[{progress_tag}] uL_used/uR_used:", uL_px, uR_px)
				# Intrinsics/dist used (bulk path)
				print(f"[{progress_tag}] bulk path K0 fx,cx,fy,cy:", float(K0[0,0]), float(K0[0,2]), float(K0[1,1]), float(K0[1,2]), "dist0[:5]:", np.asarray(D0).ravel()[:5], "img_sz:", image_size)
				print(f"[{progress_tag}] bulk path K1 fx,cx,fy,cy:", float(K1[0,0]), float(K1[0,2]), float(K1[1,1]), float(K1[1,2]), "dist1[:5]:", np.asarray(D1).ravel()[:5], "img_sz:", image_size)
				# DLT triangulation (numpy only)
				def triangulate_dlt(uL: np.ndarray, uR: np.ndarray, K_L: np.ndarray, K_R: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
					P0 = K_L @ np.hstack([np.eye(3), np.zeros((3, 1))])
					P1 = K_R @ np.hstack([R, t.reshape(3, 1)])
					xL = np.array([uL[0], uL[1], 1.0], dtype=np.float64)
					xR = np.array([uR[0], uR[1], 1.0], dtype=np.float64)
					A = np.stack([
						xL[0] * P0[2] - P0[0],
						xL[1] * P0[2] - P0[1],
						xR[0] * P1[2] - P1[0],
						xR[1] * P1[2] - P1[1],
					], axis=0)
					_, _, Vt = np.linalg.svd(A)
					Xh = Vt[-1]
					X = (Xh[:3] / max(1e-12, Xh[3])).astype(np.float64)
					return X
				X_truth = triangulate_dlt(uL_px, uR_px, K0.astype(np.float64), K1.astype(np.float64), Rst.astype(np.float64), Tst.astype(np.float64))
				X_state = X_init[DBG_F, dbg_j].astype(np.float64)
				use_truth = True
				X_used = X_truth if use_truth else X_state
				print(f"[{progress_tag}] X_check_source:", "truth_X" if use_truth else "state_X")
				print(f"[{progress_tag}] X_truth:", X_truth, "X_state:", X_state, "||state-truth||:", float(np.linalg.norm(X_state - X_truth)) if np.all(np.isfinite(X_state)) else float("nan"))
				# Project back via same pinhole (optimizer path)
				X_L_raw = X_used
				X_R_raw = (Rst @ X_L_raw + Tst.reshape(3)).astype(np.float64)
				P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
				P1 = K1 @ np.hstack([Rst.astype(np.float64), Tst.reshape(3, 1).astype(np.float64)])
				def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
					Xh = np.append(X, 1.0)
					x = P @ Xh
					return (x[:2] / max(1e-12, x[2])).astype(np.float64)
				uL_hat = project(P0, X_L_raw)
				uR_hat = project(P1, X_L_raw)
				duL = uL_hat - uL_px
				duR = uR_hat - uR_px
				eL_sq = float(np.sum(duL ** 2))
				eR_sq = float(np.sum(duR ** 2))
				eL = float(np.sqrt(eL_sq))
				eR = float(np.sqrt(eR_sq))
				# Raw X/Z diagnostics
				isfinite_XL = bool(np.isfinite(X_L_raw).all())
				zL_raw = float(X_L_raw[2])
				zR_raw = float(X_R_raw[2])
				zL_clamped = float(max(1e-12, zL_raw))
				zR_clamped = float(max(1e-12, zR_raw))
				print(f"[{progress_tag}] X_L_raw:", X_L_raw, "isfinite(X_L_raw):", isfinite_XL, "||X_L||:", float(np.linalg.norm(X_L_raw)), "zL_raw:", zL_raw, "zL_clamped:", zL_clamped)
				print(f"[{progress_tag}] X_R_raw:", X_R_raw, "||X_R||:", float(np.linalg.norm(X_R_raw)), "zR_raw:", zR_raw, "zR_clamped:", zR_clamped)
				# Projection diagnostics
				isfinite_uLhat = bool(np.isfinite(uL_hat).all())
				isfinite_uRhat = bool(np.isfinite(uR_hat).all())
				print(f"[{progress_tag}] uL_obs:", uL_px, "uL_hat:", uL_hat, "isfinite(uL_hat):", isfinite_uLhat, "duL:", duL, "sq:", eL_sq, "l2:", eL)
				print(f"[{progress_tag}] uR_obs:", uR_px, "uR_hat:", uR_hat, "isfinite(uR_hat):", isfinite_uRhat, "duR:", duR, "sq:", eR_sq, "l2:", eR)
				# Mask booleans used by loss (approximate)
				depth_min = 0.2
				m2dL = bool(validL[DBG_F, dbg_j])
				m2dR = bool(validR[DBG_F, dbg_j])
				# Components behind m2d (conf and thresholds, in-bounds, finite)
				conf_thr = float(args.conf_min)
				print(f"[{progress_tag}] m2d components: confL/confR:", float(L['conf'][DBG_F, dbg_j]), float(R['conf'][DBG_F, dbg_j]), "thr:", conf_thr)
				Wn, Hn = int(image_size[0]), int(image_size[1])
				in_bounds_L = bool((0 <= uL_px[0] < Wn) and (0 <= uL_px[1] < Hn))
				in_bounds_R = bool((0 <= uR_px[0] < Wn) and (0 <= uR_px[1] < Hn))
				print(f"[{progress_tag}] in_bounds L/R:", in_bounds_L, in_bounds_R, "finite(u_used) L/R:", bool(np.isfinite(uL_px).all()), bool(np.isfinite(uR_px).all()))
				m_depthL = bool(zL_raw > depth_min)
				m_depthR = bool(zR_raw > depth_min)
				m_finL = isfinite_uLhat
				m_finR = isfinite_uRhat
				final_maskL = m2dL and m_depthL and m_finL
				final_maskR = m2dR and m_depthR and m_finR
				print(f"[{progress_tag}] masks: m2dL={m2dL} m2dR={m2dR} m_depthL={m_depthL} m_depthR={m_depthR} m_finL={m_finL} m_finR={m_finR} finalL={final_maskL} finalR={final_maskR}")
			except Exception as e:
				print(f"[{progress_tag}] OPT PATH CHECK skipped:", e)
		X_opt_w, scale_w, losses_w = optimize_joints3d_with_priors(
			t=t[s:end],
			kptsL_px=kL_f[s:end],
			confL=L["conf"][s:end],
			kptsR_px=kR_f[s:end],
			confR=R["conf"][s:end],
			validL=validL[s:end],
			validR=validR[s:end],
			K_L=K0,
			K_R=K1,
			R=Rst,
			t_vec=Tst,
			joint_names=joint_names,
			X_init=X_init[s:end],
			valid3d_init=valid3d_init[s:end],
			cfg=OptimizeConfig(
				log_every=50,
				print_progress=True,
				lambda_bone=float(args.lambda_bone),
				lambda_acc=float(args.lambda_acc),
				# Keep velocity weak to emphasize constant acceleration primarily
				lambda_vel=1e-6,
			),
			progress_prefix=progress_tag,
		)
		# Accumulate
		X_accum[s:end] += X_opt_w.astype(np.float64)
		N_accum[s:end] += 1.0
		wlen = float(end - s)
		scale_weighted_sum += scale_w * wlen
		scale_weight += wlen
		last_losses = losses_w
		# Window-level RMSE summary
		_, _, ovL_w, ovR_w = rmse_reprojection_per_joint(
			X_opt_w, kL_f[s:end], kR_f[s:end], validL[s:end], validR[s:end], K0, K1, Rst, Tst
		)
		print(
			f"[{progress_tag}] opt overall L(mean/med/p95)={ovL_w['mean']:.2f}/{ovL_w['median']:.2f}/{ovL_w['p95']:.2f} "
			f"R={ovR_w['mean']:.2f}/{ovR_w['median']:.2f}/{ovR_w['p95']:.2f} px, s={scale_w:.4f}"
		)
	# Combine overlapping windows (average)
	with np.errstate(invalid="ignore", divide="ignore"):
		X_opt = np.where(N_accum > 0, (X_accum / np.maximum(N_accum, 1e-8)), X_init.astype(np.float64)).astype(np.float32)
	scale = float(scale_weighted_sum / max(1.0, scale_weight))
	losses = last_losses

	# Reproject optimized and init 3D (undistorted pixel space, P=K)
	def _project_np(X: np.ndarray, K: np.ndarray) -> np.ndarray:
		x = X @ K.T  # (...,3)
		den = x[..., 2]
		den = np.where(np.abs(den) > 1e-12, den, np.where(den >= 0.0, 1e-12, -1e-12))
		return (x[..., :2] / den[..., None]).astype(np.float32)

	X_opt_L = X_opt.astype(np.float64)
	X_opt_R = (X_opt_L @ Rst.T) + Tst.reshape(1, 1, 3)
	reproj_opt_left = _project_np(X_opt_L, K0.astype(np.float64))
	reproj_opt_right = _project_np(X_opt_R, K1.astype(np.float64))

	X_init_L = X_init.astype(np.float64)
	X_init_R = (X_init_L @ Rst.T) + Tst.reshape(1, 1, 3)
	reproj_init_left = _project_np(X_init_L, K0.astype(np.float64))
	reproj_init_right = _project_np(X_init_R, K1.astype(np.float64))

	# RMSE before/after
	rmse_init_L, rmse_init_R, overall_init_L, overall_init_R = rmse_reprojection_per_joint(
		np.nan_to_num(X_init, nan=0.0), kL_f, kR_f, validL, validR, K0, K1, Rst, Tst
	)
	rmse_opt_L, rmse_opt_R, overall_opt_L, overall_opt_R = rmse_reprojection_per_joint(
		X_opt, kL_f, kR_f, validL, validR, K0, K1, Rst, Tst
	)

	# Rig summary (optional metadata)
	baseline_m = float(np.linalg.norm(Tst.reshape(3)))
	rig_summary = {
		"image_size": np.array(image_size, dtype=np.int32),
		"K_left": K0,
		"D_left": D0,
		"K_right": K1,
		"D_right": D1,
		"R": Rst,
		"T": Tst,
		"baseline_m": baseline_m,
		"undistorted_px": True,
		"K_left_hash": _hash_array(K0),
		"K_right_hash": _hash_array(K1),
		"R_hash": _hash_array(Rst),
		"T_hash": _hash_array(Tst),
	}

	# Choose output path
	if args.out_npz:
		out_npz = Path(args.out_npz)
	else:
		default_dir = Path(args.left_npz).parent
		out_npz = default_dir / "pose3d_optimized.npz"

	save_npz_compressed(
		out_npz,
		t=t.astype(np.float32),
		frame_idx=frame_idx.astype(np.int32),
		joint_names=np.array(joint_names, dtype=object),
		joints3d=X_opt.astype(np.float32),
		valid2d_left=validL.astype(bool),
		valid2d_right=validR.astype(bool),
		rejected2d_left=kfL["was_rejected"].astype(bool),
		rejected2d_right=kfR["was_rejected"].astype(bool),
		joints3d_init=X_init.astype(np.float32),
		valid3d_init=valid3d_init.astype(bool),
		scale=np.array([scale], dtype=np.float32),
		rig_summary=np.array([rig_summary], dtype=object),
		rmse_init_left=rmse_init_L.astype(np.float32),
		rmse_init_right=rmse_init_R.astype(np.float32),
		rmse_opt_left=rmse_opt_L.astype(np.float32),
		rmse_opt_right=rmse_opt_R.astype(np.float32),
		rmse_overall_init_left=np.array([overall_init_L], dtype=object),
		rmse_overall_init_right=np.array([overall_init_R], dtype=object),
		rmse_overall_opt_left=np.array([overall_opt_L], dtype=object),
		rmse_overall_opt_right=np.array([overall_opt_R], dtype=object),
		# Reprojections
		reproj_opt_left=reproj_opt_left,
		reproj_opt_right=reproj_opt_right,
		reproj_init_left=reproj_init_left,
		reproj_init_right=reproj_init_right,
		# Provide explicit loss names users asked for
		losses=np.array([
			{
				"reproj_L": losses.get("E_reproj_L", np.nan),
				"reproj_R": losses.get("E_reproj_R", np.nan),
				"bone": losses.get("E_bone", np.nan),
				"z_barrier": losses.get("E_zbarrier", 0.0),
				"scale_prior": losses.get("E_scale", np.nan),
				"vel": losses.get("E_vel", np.nan),
				"acc": losses.get("E_acc", np.nan),
				"init": losses.get("E_init", np.nan),
			}
		], dtype=object),
	)

	# Console report
	J = len(joint_names)
	print(f"[OK] Saved {out_npz}")
	print(
		f"Overall reprojection (px): "
		f"init L(mean/med/p95)={overall_init_L['mean']:.2f}/{overall_init_L['median']:.2f}/{overall_init_L['p95']:.2f} "
		f"R(mean/med/p95)={overall_init_R['mean']:.2f}/{overall_init_R['median']:.2f}/{overall_init_R['p95']:.2f}  |  "
		f"opt L(mean/med/p95)={overall_opt_L['mean']:.2f}/{overall_opt_L['median']:.2f}/{overall_opt_L['p95']:.2f} "
		f"R(mean/med/p95)={overall_opt_R['mean']:.2f}/{overall_opt_R['median']:.2f}/{overall_opt_R['p95']:.2f}"
	)
	for j in range(J):
		name = joint_names[j]
		print(
			f"  {name:>18s} | init RMSE L/R: {rmse_init_L[j]:6.2f}/{rmse_init_R[j]:6.2f}  ->  "
			f"opt RMSE L/R: {rmse_opt_L[j]:6.2f}/{rmse_opt_R[j]:6.2f} px"
		)


if __name__ == "__main__":
	main()


