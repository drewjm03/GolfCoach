from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

import numpy as np
import cv2

from golfcoach.io.npz_io import load_npz
from golfcoach.io.rig_config import load_rig_config
from golfcoach.pose3d.joints3d_kf_triang_opt import undistort_kpts_pixels


def skew(t: np.ndarray) -> np.ndarray:
	t = t.reshape(3)
	return np.array(
		[
			[0.0, -t[2], t[1]],
			[t[2], 0.0, -t[0]],
			[-t[1], t[0], 0.0],
		],
		dtype=np.float64,
	)


def fundamental_from_KRT(K0: np.ndarray, K1: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
	# F = K1^-T [t]_x R K0^-1
	K0i = np.linalg.inv(K0)
	K1iT = np.linalg.inv(K1).T
	E = skew(t) @ R
	return K1iT @ E @ K0i


def point_line_distance(l: np.ndarray, x: np.ndarray) -> float:
	# l: (3,), x: (3,) homogeneous
	return float(abs(l @ x) / max(1e-12, np.sqrt(l[0] * l[0] + l[1] * l[1])))


def triangulate_dlt(uL: np.ndarray, uR: np.ndarray, P0: np.ndarray, P1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	xL = np.array([uL[0], uL[1], 1.0], dtype=np.float64)
	xR = np.array([uR[0], uR[1], 1.0], dtype=np.float64)
	A = np.stack(
		[
			xL[0] * P0[2] - P0[0],
			xL[1] * P0[2] - P0[1],
			xR[0] * P1[2] - P1[0],
			xR[1] * P1[2] - P1[1],
		],
		axis=0,
	)
	_, _, Vt = np.linalg.svd(A)
	Xh = Vt[-1]
	w = float(Xh[3])
	den = w if abs(w) > 1e-12 else (1e-12 if w >= 0 else -1e-12)
	X = Xh[:3] / den
	return X.astype(np.float64), Xh.astype(np.float64)


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
	Xh = np.append(X, 1.0)
	x = P @ Xh
	den = float(x[2])
	den = den if abs(den) > 1e-12 else (1e-12 if den >= 0 else -1e-12)
	return (x[:2] / den).astype(np.float64)


def _align_by_frame_idx(left: dict, right: dict) -> tuple[
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	list[str],
	tuple[int, int],
]:
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
	for ti, fi in enumerate(common_idx):
		li = mapL[fi]
		ri = mapR[fi]
		t[ti] = float(tL[li])
		kptsL[ti] = kL[li]
		kptsR[ti] = kR[ri]
		confL[ti] = cL[li]
		confR[ti] = cR[ri]

	return t, kptsL, kptsR, confL, confR, jL, szL


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--left_npz", required=True)
	ap.add_argument("--right_npz", required=True)
	ap.add_argument("--rig_json", required=True)
	ap.add_argument("--ti", type=int, required=True)
	ap.add_argument("--conf_min", type=float, default=0.05)
	ap.add_argument("--flip_right_x", action="store_true")
	args = ap.parse_args()

	left = load_npz(args.left_npz)
	right = load_npz(args.right_npz)

	# Align by frame_idx intersection
	t, kL, kR, cL, cR, joint_names, image_size = _align_by_frame_idx(left, right)
	W, H = int(image_size[0]), int(image_size[1])

	img_sz_rig, K0, D0, K1, D1, R, T = load_rig_config(args.rig_json)
	# Scale intrinsics if rig's image_size differs
	Wc, Hc = int(img_sz_rig[0]), int(img_sz_rig[1])
	if (Wc, Hc) != (W, H):
		sx = float(W) / float(Wc) if Wc > 0 else 1.0
		sy = float(H) / float(Hc) if Hc > 0 else 1.0
		K0 = K0.copy().astype(np.float64)
		K1 = K1.copy().astype(np.float64)
		K0[0, 0] *= sx; K0[0, 2] *= sx; K0[1, 1] *= sy; K0[1, 2] *= sy
		K1[0, 0] *= sx; K1[0, 2] *= sx; K1[1, 1] *= sy; K1[1, 2] *= sy

	# Undistort to pixel space using P=K
	kL_ud = undistort_kpts_pixels(kL, K0, D0)
	kR_ud = undistort_kpts_pixels(kR, K1, D1)

	ti = int(args.ti)
	if not (0 <= ti < kL_ud.shape[0]):
		raise ValueError(f"ti out of range: {ti} (T={kL_ud.shape[0]})")

	P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
	P1 = K1 @ np.hstack([R.astype(np.float64), T.reshape(3, 1).astype(np.float64)])

	F = fundamental_from_KRT(K0.astype(np.float64), K1.astype(np.float64), R.astype(np.float64), T.astype(np.float64))

	ok = rz = rr = tinyw = 0
	print(f"=== frame ti={ti} (W,H)=({W},{H}) flip_right_x={args.flip_right_x} ===")
	print("K0 fx,cx,fy,cy:", float(K0[0, 0]), float(K0[0, 2]), float(K0[1, 1]), float(K0[1, 2]))
	print("K1 fx,cx,fy,cy:", float(K1[0, 0]), float(K1[0, 2]), float(K1[1, 1]), float(K1[1, 2]))

	for j, name in enumerate(joint_names):
		if not (cL[ti, j] >= args.conf_min and cR[ti, j] >= args.conf_min):
			continue
		uL = kL_ud[ti, j].astype(np.float64)
		uR = kR_ud[ti, j].astype(np.float64)
		if args.flip_right_x:
			uR = np.array([(W - 1) - uR[0], uR[1]], dtype=np.float64)

		# Epipolar distance (right point to line from left)
		xL = np.array([uL[0], uL[1], 1.0], dtype=np.float64)
		xR = np.array([uR[0], uR[1], 1.0], dtype=np.float64)
		lR = F @ xL
		epi_d = point_line_distance(lR, xR)

		X, Xh = triangulate_dlt(uL, uR, P0, P1)
		w = float(Xh[3])
		XR = (R @ X + T.reshape(3)).astype(np.float64)
		zL, zR = float(X[2]), float(XR[2])

		uL_hat = project(P0, X)
		uR_hat = project(P1, X)
		eL = float(np.linalg.norm(uL_hat - uL))
		eR = float(np.linalg.norm(uR_hat - uR))

		disp = float(uL[0] - uR[0])
		tiny_w = abs(w) < 1e-8

		status = "OK"
		if tiny_w:
			status = "TINY_W"
			tinyw += 1
		if not (zL > 0.2 and zR > 0.2):
			status = "REJECT_Z"
			rz += 1
		# Optional reprojection reject (commented by default)
		# if eL > 20 or eR > 20: status = "REJECT_REPROJ"; rr += 1

		if status == "OK":
			ok += 1

		print(
			f"{j:02d} {name:>16s} {status:>10s} "
			f"uL_used={uL} uR_used={uR} "
			f"disp={disp:+7.2f} epi={epi_d:7.3f} w={w:+.3e} "
			f"X={X} zL/zR={zL:+.3f}/{zR:+.3f} eL/eR={eL:6.2f}/{eR:6.2f}"
		)

	print(f"--- summary --- ok={ok} reject_z={rz} reject_reproj={rr} tiny_w={tinyw}")


if __name__ == "__main__":
	main()


