from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.optim import Adam


# ------------------------------
# 2D undistortion
# ------------------------------

def safe_div(num: np.ndarray, den: float, eps: float = 1e-12) -> np.ndarray:
	den = float(den)
	if abs(den) < eps:
		den = eps if den >= 0.0 else -eps
	return num / den


def undistort_kpts_pixels(kpts_px: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
	"""
	kpts_px: (T,J,2) distorted pixel coords
	returns: (T,J,2) undistorted pixel coords (same pixel coordinate system), using P=K
	"""
	if kpts_px.ndim != 3 or kpts_px.shape[-1] != 2:
		raise ValueError(f"Expected kpts_px shape (T,J,2), got {kpts_px.shape}")
	T, J, _ = kpts_px.shape
	pts = kpts_px.reshape(T * J, 1, 2).astype(np.float64)
	undist = cv2.undistortPoints(pts, K.astype(np.float64), D.astype(np.float64), P=K.astype(np.float64))
	undist = undist.reshape(T, J, 2).astype(kpts_px.dtype, copy=False)
	return undist


# ------------------------------
# Kalman filter with gating (per joint)
# ------------------------------

@dataclass
class Kalman2DState:
	x: np.ndarray  # (4,)
	P: np.ndarray  # (4,4)
	initialized: bool


def _kf_mats(dt: float, q_accel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	# State: [u, v, du, dv], Measurement: [u, v]
	F = np.array(
		[
			[1.0, 0.0, dt, 0.0],
			[0.0, 1.0, 0.0, dt],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float64,
	)
	H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
	dt2 = dt * dt
	dt3 = dt2 * dt
	dt4 = dt2 * dt2
	Q = q_accel * np.array(
		[
			[dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
			[0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
			[dt3 / 2.0, 0.0, dt2, 0.0],
			[0.0, dt3 / 2.0, 0.0, dt2],
		],
		dtype=np.float64,
	)
	return F, H, Q


def _conf_to_sigma_px(conf: float, sigma_min_px: float, sigma_max_px: float) -> float:
	c = float(np.clip(conf, 0.0, 1.0))
	return sigma_min_px + (sigma_max_px - sigma_min_px) * (1.0 - c)


def kalman_filter_gating_2d(
	t: np.ndarray,              # (T,)
	kpts: np.ndarray,           # (T,J,2) undistorted px
	conf: np.ndarray,           # (T,J)
	conf_min: float = 0.05,
	sigma_min_px: float = 2.0,
	sigma_max_px: float = 40.0,
	q_accel: float = 50.0,
	patch_gate_thresh: float | None = None,
	gate_thresh: float = 9.21,
) -> Dict[str, np.ndarray]:
	"""
	Returns dict with kpts_filt, valid, was_rejected, d2
	"""
	if t.ndim != 1:
		raise ValueError("t must be (T,)")
	if kpts.ndim != 3 or kpts.shape[-1] != 2:
		raise ValueError("kpts must be (T,J,2)")
	if conf.shape != kpts.shape[:2]:
		raise ValueError("conf must be (T,J)")

	T, J, _ = kpts.shape
	kpts_filt = np.copy(kpts).astype(np.float64)
	valid = np.zeros((T, J), dtype=bool)
	was_rejected = np.zeros((T, J), dtype=bool)
	d2 = np.full((T, J), np.nan, dtype=np.float64)

	# Per-joint KF state
	states: List[Kalman2DState] = [
		Kalman2DState(
			x=np.zeros(4, dtype=np.float64),
			P=np.diag([100.0**2, 100.0**2, 50.0**2, 50.0**2]).astype(np.float64),
			initialized=False,
		)
		for _ in range(J)
	]

	for ti in range(T):
		dt = float(t[ti] - t[ti - 1]) if ti > 0 else 0.0
		if not np.isfinite(dt) or dt < 0:
			dt = 0.0
		F, H, Q = _kf_mats(dt, q_accel)

		for j in range(J):
			state = states[j]
			z = kpts[ti, j, :].astype(np.float64)
			c = float(conf[ti, j])
			has_meas = np.all(np.isfinite(z))
			accept_meas = False

			if not state.initialized:
				if has_meas and c >= conf_min:
					state.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
					state.P = np.diag([100.0**2, 100.0**2, 50.0**2, 50.0**2]).astype(np.float64)
					state.initialized = True
					accept_meas = True
				else:
					# Not initialized and no acceptable measurement
					continue

			# Predict
			x_pred = F @ state.x
			P_pred = F @ state.P @ F.T + Q

			if has_meas and c >= conf_min:
				sigma_px = _conf_to_sigma_px(c, sigma_min_px, sigma_max_px)
				R = (sigma_px ** 2) * np.eye(2, dtype=np.float64)
				y = z - (H @ x_pred)
				S = H @ P_pred @ H.T + R
				try:
					S_inv = np.linalg.inv(S)
				except np.linalg.LinAlgError:
					S_inv = np.linalg.pinv(S)
				metric = float(y.T @ S_inv @ y)
				d2[ti, j] = metric
				if metric <= gate_thresh:
					# Update
					K_gain = P_pred @ H.T @ S_inv
					x_post = x_pred + K_gain @ y
					P_post = (np.eye(4) - K_gain @ H) @ P_pred
					state.x, state.P = x_post, P_post
					accept_meas = True
					valid[ti, j] = True
					kpts_filt[ti, j, :] = (H @ x_post).astype(np.float64)
				else:
					# Gate out
					was_rejected[ti, j] = True
					state.x, state.P = x_pred, P_pred
					kpts_filt[ti, j, :] = (H @ x_pred).astype(np.float64)
			else:
				# No measurement: just propagate
				state.x, state.P = x_pred, P_pred
				kpts_filt[ti, j, :] = (H @ x_pred).astype(np.float64)

	return {
		"kpts_filt": kpts_filt.astype(np.float32),
		"valid": valid,
		"was_rejected": was_rejected,
		"d2": d2.astype(np.float32),
	}


# ------------------------------
# Triangulation
# ------------------------------

def triangulate_init(
	kptsL: np.ndarray, kptsR: np.ndarray,   # (T,J,2) undistorted px
	validL: np.ndarray, validR: np.ndarray, # (T,J)
	K_L: np.ndarray, K_R: np.ndarray,
	R: np.ndarray, t: np.ndarray,
	z_min: float = 0.2,
	max_dist_m: float = 20.0,
	reproj_thresh_px: float = 20.0,
	joint_names: Optional[Sequence[str]] = None,
	dbg_ti: Optional[int] = None,
	dbg_joint_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
	if kptsL.shape != kptsR.shape or kptsL.ndim != 3 or kptsL.shape[-1] != 2:
		raise ValueError("kptsL/kptsR must be (T,J,2) and same shape")
	if validL.shape != kptsL.shape[:2] or validR.shape != kptsR.shape[:2]:
		raise ValueError("validL/validR must be (T,J)")
	T, J, _ = kptsL.shape

	P_L = K_L @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
	P_R = K_R @ np.hstack([R.astype(np.float64), t.reshape(3, 1).astype(np.float64)])

	X_out = np.full((T, J, 3), np.nan, dtype=np.float64)
	valid3d = np.zeros((T, J), dtype=bool)
	counts = dict(both2d=0, tri_ok=0, reject_z=0, reject_norm=0, reject_reproj=0)

	# Quick range check to catch normalized vs pixel mismatch
	try:
		uL_min, uL_max = float(np.nanmin(kptsL)), float(np.nanmax(kptsL))
		uR_min, uR_max = float(np.nanmin(kptsR)), float(np.nanmax(kptsR))
		print(f"uL_used range: {uL_min:.3f} .. {uL_max:.3f} | uR_used range: {uR_min:.3f} .. {uR_max:.3f}")
	except Exception:
		pass

	dbg_j: Optional[int] = None
	if (dbg_joint_name is not None) and (joint_names is not None):
		try:
			dbg_j = find_joint(joint_names, [dbg_joint_name])
		except Exception:
			dbg_j = None

	def _triangulate_dlt(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
		P0 = P_L
		P1 = P_R
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
		# Enforce positive homogeneous scale
		if float(Xh[3]) < 0.0:
			Xh = -Xh
		w = float(Xh[3])
		den = w if abs(w) > 1e-12 else 1e-12
		X = (Xh[:3] / den).astype(np.float64)
		return X

	for ti in range(T):
		for j in range(J):
			if not (validL[ti, j] and validR[ti, j]):
				continue
			counts["both2d"] += 1
			uL = kptsL[ti, j, :].astype(np.float64)
			uR = kptsR[ti, j, :].astype(np.float64)
			# Debug print for a specific frame/joint
			if dbg_ti is not None and ti == dbg_ti and dbg_j is not None and j == dbg_j:
				print("\n=== BULK DEBUG frame/joint ===")
				print("uL_used:", uL, "uR_used:", uR)
				print(
					"K_L fx,cx,fy,cy:",
					float(K_L[0, 0]),
					float(K_L[0, 2]),
					float(K_L[1, 1]),
					float(K_L[1, 2]),
				)
				print(
					"K_R fx,cx,fy,cy:",
					float(K_R[0, 0]),
					float(K_R[0, 2]),
					float(K_R[1, 1]),
					float(K_R[1, 2]),
				)
				print("||t||:", float(np.linalg.norm(t)), "t:", t)
				print("R[0]:", R[0])
			# Triangulate with DLT (numpy only)
			X = _triangulate_dlt(uL, uR)
			# Depth checks in both cams
			X_L = X
			X_R = (R @ X_L + t.reshape(3)).astype(np.float64)
			if not (np.isfinite(X_L).all() and np.isfinite(X_R).all()):
				continue
			reject_z = not (X_L[2] > z_min and X_R[2] > z_min)
			reject_norm = np.linalg.norm(X_L) > max_dist_m
			# Reprojection check
			xL = (K_L @ X_L)
			xR = (K_R @ X_R)
			uL_hat = safe_div(xL[:2], xL[2], eps=1e-8)
			uR_hat = safe_div(xR[:2], xR[2], eps=1e-8)
			errL = float(np.linalg.norm(uL_hat - uL))
			errR = float(np.linalg.norm(uR_hat - uR))
			reject_reproj = (errL > reproj_thresh_px) or (errR > reproj_thresh_px)
			if dbg_ti is not None and ti == dbg_ti and dbg_j is not None and j == dbg_j:
				print("X:", X_L, "zL/zR:", float(X_L[2]), float(X_R[2]), "eL/eR:", errL, errR)
				print("reject_z/reject_reproj/reject_norm:", reject_z, reject_reproj, reject_norm)
			if reject_z:
				counts["reject_z"] += 1
				continue
			if reject_norm:
				counts["reject_norm"] += 1
				continue
			if reject_reproj:
				counts["reject_reproj"] += 1
				continue
			X_out[ti, j, :] = X_L
			valid3d[ti, j] = True
			counts["tri_ok"] += 1

	return X_out.astype(np.float32), valid3d, counts


# ------------------------------
# Bone priors (Plagenhoef)
# ------------------------------

def find_joint(joint_names: Sequence[str], candidates: Sequence[str]) -> int:
	# Prefer exact (case-insensitive) match, then fallback to substring.
	jn_lower = [str(s).lower() for s in joint_names]
	cands_lower = [str(c).lower() for c in candidates]
	# Exact first
	for cl in cands_lower:
		for i, nm in enumerate(jn_lower):
			if cl == nm:
				return i
	# Substring fallback
	for cl in cands_lower:
		for i, nm in enumerate(jn_lower):
			if cl in nm:
				return i
	raise KeyError(f"Could not find any candidate {list(candidates)} in joint_names={list(joint_names)}")


def build_bone_edges_for_height_5ft10(joint_names: Sequence[str]) -> List[Tuple[int, int, float]]:
	"""
	Returns list of (a_idx, b_idx, L_prior_meters) using Plagenhoef percentages and height=1.778m.
	"""
	H = 1.778  # meters (5'10")
	L_upper_arm = 0.172 * H  # 0.306 m
	L_forearm = 0.157 * H    # 0.279 m
	L_thigh = 0.232 * H      # 0.412 m
	L_shank = 0.247 * H      # 0.439 m
	L_trunk = 0.300 * H      # 0.533 m (mid_hip -> mid_shoulder)
	L_shoulder_breadth = 0.245 * H  # 0.436 m

	idx_L_sh = find_joint(joint_names, ["left_shoulder", "l_shoulder"])
	idx_R_sh = find_joint(joint_names, ["right_shoulder", "r_shoulder"])
	idx_L_el = find_joint(joint_names, ["left_elbow", "l_elbow"])
	idx_R_el = find_joint(joint_names, ["right_elbow", "r_elbow"])
	idx_L_wr = find_joint(joint_names, ["left_wrist", "l_wrist"])
	idx_R_wr = find_joint(joint_names, ["right_wrist", "r_wrist"])
	idx_L_hi = find_joint(joint_names, ["left_hip", "l_hip"])
	idx_R_hi = find_joint(joint_names, ["right_hip", "r_hip"])
	idx_L_kn = find_joint(joint_names, ["left_knee", "l_knee"])
	idx_R_kn = find_joint(joint_names, ["right_knee", "r_knee"])
	# Support datasets that name "ankle" as "foot"
	idx_L_an = find_joint(joint_names, ["left_foot", "l_foot", "left_ankle", "l_ankle"])
	idx_R_an = find_joint(joint_names, ["right_foot", "r_foot", "right_ankle", "r_ankle"])
	# Midpoints (hips and shoulders)
	# We'll implement trunk using midpoints on the fly in optimization

	edges: List[Tuple[int, int, float]] = []
	# Upper arms
	edges.append((idx_L_sh, idx_L_el, L_upper_arm))
	edges.append((idx_R_sh, idx_R_el, L_upper_arm))
	# Forearms
	edges.append((idx_L_el, idx_L_wr, L_forearm))
	edges.append((idx_R_el, idx_R_wr, L_forearm))
	# Thighs
	edges.append((idx_L_hi, idx_L_kn, L_thigh))
	edges.append((idx_R_hi, idx_R_kn, L_thigh))
	# Shanks
	edges.append((idx_L_kn, idx_L_an, L_shank))
	edges.append((idx_R_kn, idx_R_an, L_shank))
	# Shoulder breadth (optional but recommended)
	edges.append((idx_L_sh, idx_R_sh, L_shoulder_breadth))

	# For trunk length we will handle with midpoints (mid_hip -> mid_shoulder)
	# represented via (idx_L/R) pairs in the loss.
	return edges


# ------------------------------
# Projection helpers (undistorted pixel space)
# ------------------------------

def project_points_px(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
	# X: (...,3) -> (...,2) pixels
	x = torch.matmul(X, K.T)  # (...,3)
	u = x[..., 0] / torch.clamp_min(x[..., 2], 1e-8)
	v = x[..., 1] / torch.clamp_min(x[..., 2], 1e-8)
	return torch.stack([u, v], dim=-1)


def _project_left_right_px(
	X: torch.Tensor, K_L: torch.Tensor, K_R: torch.Tensor, R: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
	# Left frame
	xL = project_points_px(X, K_L)
	# Right frame: X_R = R @ X_L + t
	XR = torch.matmul(X, R.T) + t.view(1, 1, 3)
	xR = project_points_px(XR, K_R)
	return xL, xR


# ------------------------------
# Optimization
# ------------------------------

@dataclass
class OptimizeConfig:
	lambda_reproj: float = 1.0
	lambda_bone: float = 100.0
	lambda_vel: float = 1e-6
	lambda_acc: float = 1e-6
	lambda_scale: float = 1.0
	lambda_init: float = 1e-2
	adam1_iters: int = 300
	adam2_iters: int = 300
	lr1: float = 1e-2
	lr2: float = 5e-3
	grad_clip: float = 1.0
	log_every: int = 50
	print_progress: bool = True
	depth_min_m: float = 0.2
	depth_prior_fallback_m: float = 3.0


def _interp_fill_nans_per_joint(X: np.ndarray, valid: np.ndarray) -> np.ndarray:
	"""
	X: (T,J,3) with NaNs
	valid: (T,J) True where X has valid initial value
	Strategy: per joint, linear interpolate over time; nearest fill on edges.
	"""
	T, J, _ = X.shape
	out = X.copy()
	for j in range(J):
		valid_j = valid[:, j]
		if not np.any(valid_j):
			out[:, j, :] = 0.0
			continue
		t_idx = np.arange(T)
		for k in range(3):
			y = X[:, j, k]
			mask = np.isfinite(y) & valid_j
			if np.sum(mask) == 0:
				out[:, j, k] = 0.0
				continue
			yi = np.interp(t_idx, t_idx[mask], y[mask])
			# Edge fill (nearest)
			first_i = int(t_idx[mask][0])
			last_i = int(t_idx[mask][-1])
			yi[:first_i] = y[first_i]
			yi[last_i + 1 :] = y[last_i]
			out[:, j, k] = yi
	return out


def _seed_missing_with_depth_prior(
	X_init: np.ndarray, valid3d_init: np.ndarray,
	kptsL: np.ndarray, kptsR: np.ndarray,
	validL: np.ndarray, validR: np.ndarray,
	K_L: np.ndarray, K_R: np.ndarray,
	R: np.ndarray, t_vec: np.ndarray,
	joint_names: Sequence[str],
	depth_min_m: float = 0.2,
	depth_prior_fallback_m: float = 3.0,
) -> np.ndarray:
	"""
	Fill missing X_init (NaNs) by placing points on camera rays at a reasonable depth Z0.
	Choose Z0 as median z of valid torso joints in this window; fallback to 3.0m.
	"""
	T, J, _ = X_init.shape
	out = X_init.copy()
	# Determine torso joint indices (best-effort)
	torso_cands = [
		["spine"], ["thorax"], ["neck_base"],
		["left_hip", "l_hip"], ["right_hip", "r_hip"],
		["left_shoulder", "l_shoulder"], ["right_shoulder", "r_shoulder"],
	]
	torso_idx: List[int] = []
	for c in torso_cands:
		try:
			torso_idx.append(find_joint(joint_names, c))
		except Exception:
			pass
	torso_idx = sorted(set(torso_idx))
	# Compute Z0 from existing valid 3D
	z_vals = []
	for ti in range(T):
		for j in torso_idx:
			if valid3d_init[ti, j] and np.isfinite(out[ti, j, 2]) and out[ti, j, 2] > depth_min_m:
				z_vals.append(float(out[ti, j, 2]))
	Z0 = float(np.median(z_vals)) if len(z_vals) > 0 else float(depth_prior_fallback_m)
	# Precompute inverses
	KL_inv = np.linalg.inv(K_L.astype(np.float64))
	KR_inv = np.linalg.inv(K_R.astype(np.float64))
	R = R.astype(np.float64)
	t_vec = t_vec.reshape(3).astype(np.float64)
	for ti in range(T):
		for j in range(J):
			if valid3d_init[ti, j] and np.all(np.isfinite(out[ti, j, :])):
				continue
			# Try left ray if left 2D valid
			if validL[ti, j] and np.all(np.isfinite(kptsL[ti, j, :])):
				uv1 = np.array([kptsL[ti, j, 0], kptsL[ti, j, 1], 1.0], dtype=np.float64)
				dir_cam = KL_inv @ uv1
				if dir_cam[2] <= 1e-8:
					continue
				scale = Z0 / dir_cam[2]
				XL = dir_cam * scale
				if np.isfinite(XL).all() and XL[2] > depth_min_m:
					# Source-aware reprojection gating (left view only)
					xL = K_L @ XL
					uL_hat = safe_div(xL[:2], xL[2], eps=1e-12)
					eL = float(np.linalg.norm(uL_hat - uv1[:2]))
					if eL <= 20.0:
						out[ti, j, :] = XL.astype(np.float32)
						continue
			# Else try right ray and transform to left frame
			if validR[ti, j] and np.all(np.isfinite(kptsR[ti, j, :])):
				uv1 = np.array([kptsR[ti, j, 0], kptsR[ti, j, 1], 1.0], dtype=np.float64)
				dir_cam = KR_inv @ uv1
				if dir_cam[2] <= 1e-8:
					continue
				scale = Z0 / dir_cam[2]
				XR = dir_cam * scale
				XL = R.T @ (XR - t_vec)
				if np.isfinite(XL).all():
					# Check right depth only (source-aware)
					if XR[2] > depth_min_m:
						# Right-view reprojection gating only
						xR = K_R @ XR
						uR_hat = safe_div(xR[:2], xR[2], eps=1e-12)
						eR = float(np.linalg.norm(uR_hat - uv1[:2]))
						if eR <= 20.0:
							out[ti, j, :] = XL.astype(np.float32)
							continue
	# Fallback: simple interpolation if still NaN
	nan_mask = ~np.isfinite(out).all(axis=-1)
	if np.any(nan_mask):
		out = _interp_fill_nans_per_joint(out, valid3d_init)
	return out


def _huber(residuals: torch.Tensor, delta: float = 5.0) -> torch.Tensor:
	abs_r = residuals.abs()
	quadratic = 0.5 * (abs_r ** 2)
	linear = delta * (abs_r - 0.5 * delta)
	return torch.where(abs_r <= delta, quadratic, linear)


def _conf_to_sigma_tensor(conf: torch.Tensor, sigma_min_px: float = 2.0, sigma_max_px: float = 40.0) -> torch.Tensor:
	c = conf.clamp(0.0, 1.0)
	return sigma_min_px + (sigma_max_px - sigma_min_px) * (1.0 - c)


def optimize_joints3d_with_priors(
	t: np.ndarray,                               # (T,)
	kptsL_px: np.ndarray, confL: np.ndarray,     # (T,J,2), (T,J) undistorted px
	kptsR_px: np.ndarray, confR: np.ndarray,     # (T,J,2), (T,J) undistorted px
	validL: np.ndarray, validR: np.ndarray,      # (T,J), (T,J) from KF gating acceptance
	K_L: np.ndarray, K_R: np.ndarray,            # (3,3)
	R: np.ndarray, t_vec: np.ndarray,            # right-from-left extrinsics
	joint_names: Sequence[str],
	X_init: np.ndarray, valid3d_init: np.ndarray,
	cfg: OptimizeConfig | None = None,
	progress_prefix: str = "",
) -> Tuple[np.ndarray, float, Dict[str, float]]:
	if cfg is None:
		cfg = OptimizeConfig()

	T, J, _ = kptsL_px.shape
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize X
	X0 = _seed_missing_with_depth_prior(
		X_init.astype(np.float32), valid3d_init,
		kptsL_px, kptsR_px, validL, validR,
		K_L, K_R, R, t_vec,
		joint_names,
		depth_min_m=cfg.depth_min_m,
		depth_prior_fallback_m=cfg.depth_prior_fallback_m,
	)
	X_param = torch.nn.Parameter(torch.from_numpy(X0).to(device))
	# Constrained scale: s = 1 + 0.3 * tanh(raw) in [0.7, 1.3]
	raw_s = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))

	KL = torch.from_numpy(K_L.astype(np.float32)).to(device)
	KR = torch.from_numpy(K_R.astype(np.float32)).to(device)
	Rt = torch.from_numpy(R.astype(np.float32)).to(device)
	tt = torch.from_numpy(t_vec.astype(np.float32)).to(device)

	uL = torch.from_numpy(kptsL_px.astype(np.float32)).to(device)
	uR = torch.from_numpy(kptsR_px.astype(np.float32)).to(device)
	cL = torch.from_numpy(confL.astype(np.float32)).to(device)
	cR = torch.from_numpy(confR.astype(np.float32)).to(device)
	mL = torch.from_numpy(validL.astype(np.bool_)).to(device)
	mR = torch.from_numpy(validR.astype(np.bool_)).to(device)
	m3 = torch.from_numpy(valid3d_init.astype(np.bool_)).to(device)

	dt_np = np.maximum(1e-6, np.diff(t.astype(np.float32), prepend=t[0]).astype(np.float32))
	dt = torch.from_numpy(dt_np).to(device)

	# Precompute bone edges and trunk indices once
	edges = build_bone_edges_for_height_5ft10(joint_names)
	idx_L_sh = find_joint(joint_names, ["left_shoulder", "l_shoulder"])
	idx_R_sh = find_joint(joint_names, ["right_shoulder", "r_shoulder"])
	idx_L_hi = find_joint(joint_names, ["left_hip", "l_hip"])
	idx_R_hi = find_joint(joint_names, ["right_hip", "r_hip"])
	trunk_target_len = torch.tensor(0.300 * 1.778, dtype=torch.float32, device=device)

	def reprojection_and_bone_losses(X: torch.Tensor, with_scale: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		# Projections
		xL, xR = _project_left_right_px(X, KL, KR, Rt, tt)
		# Depth masks and finiteness (stability)
		XR = torch.matmul(X, Rt.T) + tt.view(1, 1, 3)
		zL = X[..., 2]
		zR = XR[..., 2]
		finL = torch.isfinite(xL).all(dim=-1)
		finR = torch.isfinite(xR).all(dim=-1)
		# Confidence to sigma (px)
		sL = _conf_to_sigma_tensor(cL).clamp(2.0, 40.0)  # (T,J)
		sR = _conf_to_sigma_tensor(cR).clamp(2.0, 40.0)  # (T,J)
		# Residual norms (px)
		resL_norm = torch.linalg.norm(xL - uL, dim=-1)  # (T,J)
		resR_norm = torch.linalg.norm(xR - uR, dim=-1)  # (T,J)
		# Normalize by sigma so Huber operates in sigma units
		rL_scaled = resL_norm / torch.clamp_min(sL, 1e-6)
		rR_scaled = resR_norm / torch.clamp_min(sR, 1e-6)
		# Masks (T,J)
		maskL = mL & finL & (zL > cfg.depth_min_m)
		maskR = mR & finR & (zR > cfg.depth_min_m)
		E_reproj_L = _huber(rL_scaled[maskL]).mean() if maskL.any() else torch.tensor(0.0, device=device)
		E_reproj_R = _huber(rR_scaled[maskR]).mean() if maskR.any() else torch.tensor(0.0, device=device)

		# Bone constraints
		if with_scale:
			s = 1.0 + 0.3 * torch.tanh(raw_s)
		else:
			s = torch.tensor(1.0, dtype=torch.float32, device=device)
		bone_terms: List[torch.Tensor] = []
		for a, b, Lm in edges:
			diff = X[:, a, :] - X[:, b, :]
			dist = torch.linalg.norm(diff, dim=-1)
			bone_terms.append(((dist - (s * Lm)) ** 2))
		# trunk via midpoints
		mid_sh = 0.5 * (X[:, idx_L_sh, :] + X[:, idx_R_sh, :])
		mid_hi = 0.5 * (X[:, idx_L_hi, :] + X[:, idx_R_hi, :])
		trunk_len = torch.linalg.norm(mid_sh - mid_hi, dim=-1)
		bone_terms.append((trunk_len - (s * trunk_target_len)) ** 2)
		E_bone = torch.mean(torch.stack(bone_terms, dim=0)) if bone_terms else torch.tensor(0.0, device=device)

		# Temporal smoothing (velocity and acceleration)
		# V_t = (X_t - X_{t-1})/dt
		dX = X[1:, :, :] - X[:-1, :, :]
		V = dX / dt[1:].view(-1, 1, 1)
		E_vel = torch.mean(V ** 2)
		# Acc
		dV = V[1:, :, :] - V[:-1, :, :]
		E_acc = torch.mean(dV ** 2)

		# Stay near triangulation where available
		Xi = torch.from_numpy(X_init.astype(np.float32)).to(device)
		if m3.any():
			E_init = torch.mean(((X - Xi)[m3]) ** 2)
		else:
			E_init = torch.tensor(0.0, device=device)

		losses = dict(
			E_reproj_L=E_reproj_L,
			E_reproj_R=E_reproj_R,
			E_bone=E_bone,
			E_vel=E_vel,
			E_acc=E_acc,
			E_init=E_init,
			E_zbarrier=torch.tensor(0.0, device=device),
			E_scale=( (1.0 + 0.3 * torch.tanh(raw_s)) - 1.0 ) ** 2 / (0.15 ** 2),
		)
		total = (
			cfg.lambda_reproj * (E_reproj_L + E_reproj_R)
			+ cfg.lambda_bone * E_bone
			+ cfg.lambda_vel * E_vel
			+ cfg.lambda_acc * E_acc
			+ cfg.lambda_scale * losses["E_scale"]
			+ cfg.lambda_init * E_init
		)
		return total, losses

	# Phase 1: optimize X only
	for p in [X_param]:
		p.requires_grad_(True)
	raw_s.requires_grad_(False)
	opt = Adam([X_param], lr=cfg.lr1)
	for it in range(cfg.adam1_iters):
		opt.zero_grad(set_to_none=True)
		total, _ = reprojection_and_bone_losses(X_param, with_scale=False)
		total.backward()
		nn.utils.clip_grad_norm_([X_param], cfg.grad_clip)
		opt.step()
		if cfg.print_progress and cfg.log_every > 0 and ((it + 1) % cfg.log_every == 0 or it == cfg.adam1_iters - 1):
			with torch.no_grad():
				_, comp = reprojection_and_bone_losses(X_param, with_scale=False)
			print(
				f"[{progress_prefix}] phase1 iter {it+1}/{cfg.adam1_iters} "
				f"total={float(total.detach().cpu().item()):.6f} "
				f"reprojL={float(comp['E_reproj_L'].detach().cpu().item()):.6f} "
				f"reprojR={float(comp['E_reproj_R'].detach().cpu().item()):.6f} "
				f"bone={float(comp['E_bone'].detach().cpu().item()):.6f}"
			)

	# Phase 2: allow scale
	for p in [X_param, raw_s]:
		p.requires_grad_(True)
	opt2 = Adam([X_param, raw_s], lr=cfg.lr2)
	for it in range(cfg.adam2_iters):
		opt2.zero_grad(set_to_none=True)
		total, _ = reprojection_and_bone_losses(X_param, with_scale=True)
		total.backward()
		nn.utils.clip_grad_norm_([X_param, raw_s], cfg.grad_clip)
		opt2.step()
		if cfg.print_progress and cfg.log_every > 0 and ((it + 1) % cfg.log_every == 0 or it == cfg.adam2_iters - 1):
			with torch.no_grad():
				_, comp = reprojection_and_bone_losses(X_param, with_scale=True)
				cur_s = float((1.0 + 0.3 * torch.tanh(raw_s)).detach().cpu().item())
			print(
			 f"[{progress_prefix}] phase2 iter {it+1}/{cfg.adam2_iters} "
			 f"total={float(total.detach().cpu().item()):.6f} "
			 f"reprojL={float(comp['E_reproj_L'].detach().cpu().item()):.6f} "
			 f"reprojR={float(comp['E_reproj_R'].detach().cpu().item()):.6f} "
			 f"bone={float(comp['E_bone'].detach().cpu().item()):.6f} "
			 f"s={cur_s:.4f}"
			)

	# Final losses for diagnostics
	with torch.no_grad():
		_, losses = reprojection_and_bone_losses(X_param, with_scale=True)

	X_out = X_param.detach().cpu().numpy().astype(np.float32)
	scale = float((1.0 + 0.3 * torch.tanh(raw_s)).detach().cpu().item())
	diag = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
	return X_out, scale, diag


# ------------------------------
# RMSE diagnostics
# ------------------------------

def rmse_reprojection_per_joint(
	X: np.ndarray,
	kptsL_px: np.ndarray, kptsR_px: np.ndarray,
	validL: np.ndarray, validR: np.ndarray,
	K_L: np.ndarray, K_R: np.ndarray, R: np.ndarray, t: np.ndarray,
	z_min: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
	"""
	Returns (rmse_L[J], rmse_R[J], overall_stats_L, overall_stats_R)
	RMSEs and overall stats computed only over frames where that camera's valid2d mask is True,
	and where X has finite values, projections are finite, and depth > z_min for that camera.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	Xt = torch.from_numpy(X.astype(np.float32)).to(device)
	KL = torch.from_numpy(K_L.astype(np.float32)).to(device)
	KR = torch.from_numpy(K_R.astype(np.float32)).to(device)
	Rt = torch.from_numpy(R.astype(np.float32)).to(device)
	tt = torch.from_numpy(t.astype(np.float32)).to(device)
	xL, xR = _project_left_right_px(Xt, KL, KR, Rt, tt)
	xL = xL.detach().cpu().numpy()
	xR = xR.detach().cpu().numpy()
	XR = (X @ R.T + t.reshape(1, 1, 3)).astype(np.float32)
	T, J, _ = X.shape
	rmseL = np.zeros(J, dtype=np.float32)
	rmseR = np.zeros(J, dtype=np.float32)
	all_errs_L: List[float] = []
	all_errs_R: List[float] = []
	X_finite = np.all(np.isfinite(X), axis=-1)  # (T,J)
	for j in range(J):
		finL = np.isfinite(xL[:, j, :]).all(axis=-1)
		finR = np.isfinite(xR[:, j, :]).all(axis=-1)
		zL = X[:, j, 2]
		zR = XR[:, j, 2]
		maskL = (validL[:, j] & X_finite[:, j] & finL & (zL > z_min))
		maskR = (validR[:, j] & X_finite[:, j] & finR & (zR > z_min))
		errL2 = xL[maskL, j, :] - kptsL_px[maskL, j, :]
		errR2 = xR[maskR, j, :] - kptsR_px[maskR, j, :]
		if errL2.size > 0:
			all_errs_L.extend(np.sqrt(np.sum(errL2 ** 2, axis=-1)).tolist())
		if errR2.size > 0:
			all_errs_R.extend(np.sqrt(np.sum(errR2 ** 2, axis=-1)).tolist())
		rmseL[j] = float(np.sqrt(np.mean(np.sum(errL2 ** 2, axis=-1))) if errL2.size > 0 else np.nan)
		rmseR[j] = float(np.sqrt(np.mean(np.sum(errR2 ** 2, axis=-1))) if errR2.size > 0 else np.nan)
	def _overall(errs: List[float]) -> Dict[str, float]:
		if len(errs) == 0:
			return dict(mean=np.nan, median=np.nan, p95=np.nan)
		arr = np.asarray(errs, dtype=np.float64)
		return dict(
			mean=float(np.mean(arr)),
			median=float(np.median(arr)),
			p95=float(np.percentile(arr, 95.0)),
		)
	return rmseL, rmseR, _overall(all_errs_L), _overall(all_errs_R)



