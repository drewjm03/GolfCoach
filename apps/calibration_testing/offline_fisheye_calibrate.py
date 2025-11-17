import os, sys, json, glob, argparse, cv2
import numpy as np

try:
	from .. import config
	from ..detect import CalibrationAccumulator
except Exception:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from apps import config
	from apps.detect import CalibrationAccumulator


def _make_board():
	dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
	ids_grid = np.arange(config.TAGS_X * config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
	ids_grid = np.flipud(ids_grid)
	ids = ids_grid.reshape(-1, 1).astype(np.int32)
	return cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)


def seed_K_fisheye(W, H, hfov_deg=104.6, vfov_deg=61.6):
	hf = np.deg2rad(hfov_deg); vf = np.deg2rad(vfov_deg)
	fx = W / max(hf, 1e-6)
	fy = H / max(vf, 1e-6)
	K = np.eye(3, dtype=np.float64)
	K[0,0], K[1,1] = fx, fy
	K[0,2], K[1,2] = W*0.5, H*0.5
	return K


def fisheye_calibrate(obj_list, img_list, image_size, K_seed, check_cond=True):
	obj_std = [o.astype(np.float64, copy=False) for o in obj_list]
	img_std = [i.astype(np.float64, copy=False) for i in img_list]
	K = K_seed.copy().astype(np.float64); D = np.zeros((4,1), np.float64)
	flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
	         cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
	         cv2.fisheye.CALIB_FIX_SKEW)
	if check_cond:
		flags |= cv2.fisheye.CALIB_CHECK_COND
	crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
	rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj_std, img_std, image_size, K, D, None, None, flags=flags, criteria=crit)
	return float(rms), K, D, rvecs, tvecs


def load_recording(folder):
	with open(os.path.join(folder, "meta.json"), "r", encoding="utf-8") as f:
		meta = json.load(f)
	W, H = int(meta["image_size"][0]), int(meta["image_size"][1])
	img_list = []
	obj_list = []
	board = _make_board()
	# Build id->obj map using acc helper
	acc = CalibrationAccumulator(board, (W, H))

	frames = sorted(glob.glob(os.path.join(folder, "frame_*.json")))
	if not frames:
		raise RuntimeError("No per-frame JSON files found in folder")
	for jf in frames:
		with open(jf, "r", encoding="utf-8") as f:
			data = json.load(f)
		ids = data.get("ids") or []
		corners = data.get("corners") or []
		if not ids or not corners:
			continue
		O, I = [], []
		for tid, c in zip(ids, corners):
			tid = int(tid)
			if tid not in acc.id_to_obj:
				continue
			O.append(acc.id_to_obj[tid])
			I.append(np.asarray(c, dtype=np.float64))
		if not O:
			continue
		obj = np.concatenate(O, 0).reshape(-1,1,3).astype(np.float64)
		img = np.concatenate(I, 0).reshape(-1,1,2).astype(np.float64)
		obj_list.append(obj)
		img_list.append(img)
	return (W, H), obj_list, img_list


def main():
	parser = argparse.ArgumentParser(description="Offline fisheye calibration on recorded keyframes.")
	parser.add_argument("--folder", "-f", required=True, help="Path to data/calibration_YYYYmmdd_HHMMSS folder")
	parser.add_argument("--hfov", type=float, default=104.6)
	parser.add_argument("--vfov", type=float, default=61.6)
	args = parser.parse_args()

	image_size, obj_list, img_list = load_recording(args.folder)
	W, H = image_size
	if not obj_list:
		print("[OFFLINE] No valid frames in folder"); return
	K_seed = seed_K_fisheye(W, H, args.hfov, args.vfov)
	try:
		rms, K, D, rvecs, tvecs = fisheye_calibrate(obj_list, img_list, image_size, K_seed, check_cond=True)
	except cv2.error as e:
		print("[OFFLINE] CHECK_COND failed, retrying without check:", e)
		rms, K, D, rvecs, tvecs = fisheye_calibrate(obj_list, img_list, image_size, K_seed, check_cond=False)
	print(f"[OFFLINE] Fisheye RMS: {rms:.3f}")
	print(f"[OFFLINE] K=\n{K}\nD={D.ravel().tolist()}")

	out_json = os.path.join(args.folder, "offline_calibration.json")
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump({
			"image_size": [W, H],
			"K": K.tolist(),
			"D": D.tolist(),
			"rms": float(rms),
			"model": "fisheye_equidistant_offline"
		}, f, indent=2)
	print(f"[OFFLINE] Wrote {out_json}")


if __name__ == "__main__":
	main()



