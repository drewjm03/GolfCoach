# MONO offline calibration (pinhole/Brown–Conrady 5-coeffs)
# - Reuses mono calibration logic from stereo_cam_calibrator_offline.py
# - Processes mono keyframes from a folder created by mono_keyframe_recorder.py

import os, sys, time, json, glob, re, argparse
import numpy as np
import cv2

# ---- app imports / fallback ----
try:
	from .. import config
	from ..detect import CalibrationAccumulator
except Exception:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from apps import config  # type: ignore
	from apps.detect import CalibrationAccumulator  # type: ignore

# Import helper functions from stereo calibrator (mono section)
try:
	from .stereo_cam_calibrator_offline import (
		_has_coverage, _print_intrinsics, seed_K_pinhole, calibrate_pinhole_full,
		_save_diag_pinhole, _view_rms_pinhole
	)
except Exception:
	# Minimal fallbacks (should match stereo file behavior)
	def _has_coverage(corners, W, H, min_span=0.20):
		xs = [p[0] for ci in corners for p in ci.reshape(4,2)]
		ys = [p[1] for ci in corners for p in ci.reshape(4,2)]
		if not xs: return False
		span_x = (max(xs)-min(xs))/float(max(1.0,W))
		span_y = (max(ys)-min(ys))/float(max(1.0,H))
		return min(span_x, span_y) >= min_span

	def _print_intrinsics(K, D):
		try:
			fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
		except Exception:
			fx = fy = cx = cy = float("nan")
		dlen = (int(D.size) if D is not None else 0)
		print(f"[INTR] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  D_len={dlen}")

	def seed_K_pinhole(W, H, f_scale=1.0):
		f = float(max(W, H)) * float(f_scale)
		K = np.eye(3, dtype=np.float64)
		K[0,0], K[1,1] = f, f
		K[0,2], K[1,2] = W*0.5, H*0.5
		return K

	def calibrate_pinhole_full(obj_list, img_list, image_size, K_seed=None):
		obj_std = [o.reshape(-1, 3).astype(np.float32, copy=False) for o in obj_list]
		img_std = [i.reshape(-1, 2).astype(np.float32, copy=False) for i in img_list]
		W, H = int(image_size[0]), int(image_size[1])
		# 104.6 deg HFOV seed, 5-coeff Brown–Conrady
		hfov_deg = 104.6
		fx = (W * 0.5) / np.tan(np.deg2rad(hfov_deg * 0.5))
		fy = fx
		cx, cy = W * 0.5, H * 0.5
		K_init = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]], dtype=np.float64)
		D_init = np.zeros((5, 1), dtype=np.float64)
		flags = cv2.CALIB_USE_INTRINSIC_GUESS
		crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
		rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_std, img_std, (W, H), K_init, D_init, flags=flags, criteria=crit)
		return float(rms), K, D, rvecs, tvecs

	def _view_rms_pinhole(obj_pts, img_pts, K, D, rvec, tvec):
		proj,_ = cv2.projectPoints(obj_pts.reshape(-1,1,3).astype(np.float32), rvec, tvec, K, D)
		proj = proj.reshape(-1,2)
		err = img_pts.reshape(-1,2).astype(np.float32) - proj
		return float(np.sqrt(np.mean(np.sum(err*err, axis=1))))


def _load_mono_keyframes(keyframes_dir: str):
	"""
	Load mono keyframes. Supports either 'frame_XXX.png' or 'frame_XXX_cam0.png'.
	Returns list of tuples: (frame_num, img_path, json_path, frame_bgr, json_data)
	"""
	keyframes_dir = os.path.abspath(keyframes_dir)
	if not os.path.isdir(keyframes_dir):
		raise RuntimeError(f"[KEYFRAMES] Directory not found: {keyframes_dir}")

	candidates = sorted(glob.glob(os.path.join(keyframes_dir, "frame_*_cam0.png")))
	is_cam0_style = True
	if not candidates:
		candidates = sorted(glob.glob(os.path.join(keyframes_dir, "frame_*.png")))
		is_cam0_style = False

	if not candidates:
		raise RuntimeError(f"[KEYFRAMES] No frame images found in {keyframes_dir}")

	keyframes = []
	for p in candidates:
		if is_cam0_style:
			m = re.search(r"frame_(\d+)_cam0\.png", p)
		else:
			m = re.search(r"frame_(\d+)\.png", p)
		if not m:
			continue
		frame_num = int(m.group(1))
		json_path = os.path.join(keyframes_dir, f"frame_{frame_num:03d}.json")
		img = cv2.imread(p, cv2.IMREAD_COLOR)
		if img is None:
			print(f"[KEYFRAMES] Warning: failed to load image {p}")
			continue
		js = None
		if os.path.exists(json_path):
			try:
				with open(json_path, "r", encoding="utf-8") as f:
					js = json.load(f)
			except Exception as e:
				print(f"[KEYFRAMES] Warning: failed to load JSON {json_path}: {e}")
		keyframes.append((frame_num, p, json_path, img, js))
	print(f"[KEYFRAMES] Loaded {len(keyframes)} mono keyframes from {keyframes_dir}")
	return keyframes


def _extract_ids_corners(js):
	if js is None:
		return None, None
	ids = js.get("ids", None)
	corners = js.get("corners", None)
	if not ids or not corners or len(ids) != len(corners):
		return None, None
	ids_arr = np.array(ids, dtype=np.int32).reshape(-1, 1)
	corners_list = []
	for c in corners:
		if len(c) != 4:
			return None, None
		corners_list.append(np.array(c, dtype=np.float32).reshape(1, 4, 2))
	return corners_list, ids_arr


def main():
	parser = argparse.ArgumentParser(description="Mono offline calibration using keyframes")
	parser.add_argument("--keyframes-dir", type=str, required=True, help="Directory with mono keyframes")
	parser.add_argument("--out-json", type=str, required=True, help="Output JSON path for intrinsics")
	parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], required=True,
	                    help="Which board to use: Harvard pickle or original 8x5 grid")
	parser.add_argument("--april-pickle", type=str, default=None,
	                    help="Path to local AprilBoards.pickle (overrides network)")
	parser.add_argument("--harvard-tag-size-m", type=float, default=None,
	                    help="Tag side length in meters for Harvard board (overrides pickle/env)")
	parser.add_argument("--harvard-tag-spacing-m", type=float, default=None,
	                    help="Tag spacing (meters) for Harvard board (informational)")
	parser.add_argument("--corner-order", type=str, default=None,
	                    help="Manual corner order override as four comma-separated indices, e.g. '0,1,2,3'")
	args = parser.parse_known_args()[0]

	# Load keyframes
	print(f"[MONO] Loading keyframes from: {args.keyframes_dir}")
	keyframes = _load_mono_keyframes(args.keyframes_dir)
	if not keyframes:
		print("[MONO] No valid keyframes found; aborting.")
		return

	# Image size
	_, _, _, frame0_bgr, js0 = keyframes[0]
	H, W = frame0_bgr.shape[:2]
	image_size = (W, H)
	print(f"[MONO] Image size: {W}x{H}")

	# Build board and accumulator (reuse same logic/params as stereo)
	try:
		from .stereo_calib_plot import load_board
	except Exception:
		sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
		from apps.stereo_calib_plot import load_board  # type: ignore

	board = load_board(board_source=args.board_source,
	                   april_pickle=args.april_pickle,
	                   harvard_tag_size_m=args.harvard_tag_size_m,
	                   harvard_tag_spacing_m=args.harvard_tag_spacing_m)

	corner_order_override = None
	disable_autoreorder = False
	if args.corner_order:
		try:
			parts = [int(x.strip()) for x in args.corner_order.split(",")]
			if len(parts) == 4 and sorted(parts) == [0, 1, 2, 3]:
				corner_order_override = parts
				disable_autoreorder = True
			else:
				print(f"[WARN] Ignoring invalid --corner-order '{args.corner_order}'")
		except Exception as e:
			print(f"[WARN] Failed to parse --corner-order: {e}")

	acc = CalibrationAccumulator(board, image_size,
	                             corner_order_override=corner_order_override,
	                             disable_corner_autoreorder=disable_autoreorder)
	print("[APRIL] Backend:", acc.get_backend_name())
	print("[APRIL] Families:", acc._apriltag_family_string())

	# Collect views
	obj_list, img_list, kept = [], [], []
	drop_no_ids = drop_few = drop_cov = drop_nomap = 0
	for frame_num, img_path, json_path, frame_bgr, js in keyframes:
		corners, ids = _extract_ids_corners(js)
		if corners is None or ids is None:
			gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
			try:
				corners, ids = acc.detect(gray)
			except Exception:
				corners, ids = [], None

		if ids is None:
			drop_no_ids += 1
			continue
		if len(ids) < config.MIN_MARKERS_PER_VIEW:
			drop_few += 1
			continue
		if not _has_coverage(corners, W, H, 0.20):
			drop_cov += 1
			continue

		O, I = [], []
		for c, iv in zip(corners, ids):
			tid = int(iv[0])
			if tid not in acc.id_to_obj:
				continue
			O.append(acc.id_to_obj[tid])
			I.append(c.reshape(-1, 2))
		if O:
			obj_cat = np.concatenate(O, 0).astype(np.float64).reshape(-1, 1, 3)
			img_cat = np.concatenate(I, 0).astype(np.float64).reshape(-1, 1, 2)
			obj_list.append(obj_cat)
			img_list.append(img_cat)
			kept.append(frame_num)
		else:
			drop_nomap += 1

	if (drop_no_ids + drop_few + drop_cov + drop_nomap) > 0:
		print(f"[MONO] Filter: kept={len(kept)}  no_ids={drop_no_ids}  few={drop_few}  coverage={drop_cov}  no_map={drop_nomap}")

	if not obj_list:
		print("[MONO] Not enough valid samples after filtering; aborting.")
		return

	# Calibrate
	print("[MONO] Calibrating intrinsics...")
	K_seed = seed_K_pinhole(W, H, f_scale=1.0)
	rms, K, D, rvecs, tvecs = calibrate_pinhole_full(obj_list, img_list, image_size, K_seed)
	print(f"[MONO] RMS: {rms:.3f} (D has {D.size if D is not None else 0} coeffs)")
	_print_intrinsics(K, D)

	# Optional per-view RMS diagnostics
	if len(obj_list) >= 1:
		print("[MONO] Per-view RMS:")
		for vi, (O, I) in enumerate(zip(obj_list, img_list)):
			view_rms = _view_rms_pinhole(O, I, K, D, rvecs[vi], tvecs[vi])
			print(f"  view {vi:03d} (frame {kept[vi]}): {view_rms:.2f} px")

	# Save JSON
	out_json = os.path.abspath(args.out_json)
	os.makedirs(os.path.dirname(out_json), exist_ok=True)
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump({
			"image_size": [int(W), int(H)],
			"K": np.asarray(K, dtype=np.float64).tolist(),
			"D": np.asarray(D, dtype=np.float64).reshape(-1, 1).tolist(),
			"rms": float(rms),
			"n_views": int(len(obj_list)),
			"board_source": str(args.board_source),
			"calib_flags": int(cv2.CALIB_USE_INTRINSIC_GUESS),
			"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
		}, f, indent=2)
	print(f"[MONO] Saved intrinsics -> {out_json}")
	print("[MONO] Calibration complete.")


if __name__ == "__main__":
	main()


