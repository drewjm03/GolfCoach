import os, sys, json, glob, argparse, cv2
import numpy as np
import matplotlib.pyplot as plt

# Allow module relative imports
try:
	from .. import config
	from ..detect import CalibrationAccumulator, board_ids_safe, reorder_corners_to_board
except Exception:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from apps import config
	from apps.detect import CalibrationAccumulator, board_ids_safe, reorder_corners_to_board


def _make_board():
	dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
	ids_grid = np.arange(config.TAGS_X * config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
	ids_grid = np.flipud(ids_grid)
	ids = ids_grid.reshape(-1, 1).astype(np.int32)
	return cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)


def _load_calibration(json_path):
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	K = np.asarray(data.get("K"), dtype=np.float64).reshape(3, 3)
	D_list = data.get("D")
	if D_list is None:
		D = np.zeros((5, 1), dtype=np.float64)
	else:
		D = np.asarray(D_list, dtype=np.float64).reshape(-1, 1)
	image_size = tuple(data.get("image_size") or [0, 0])
	return K, D, image_size, data


def _auto_find_latest_diag(data_dir):
	cands = sorted(glob.glob(os.path.join(data_dir, "*calib_diag_*.png")))
	return cands[-1] if cands else None


def _detect_from_image(img_bgr, board):
	H, W = img_bgr.shape[:2]
	acc = CalibrationAccumulator(board, (W, H))
	gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	corners, ids = acc.detect(gray)
	return acc, corners, ids


def _build_correspondences(acc, corners, ids):
	if ids is None or len(ids) == 0:
		return None, None
	obj_pts = []
	img_pts = []
	for c, iv in zip(corners, ids):
		tag_id = int(iv[0])
		if tag_id not in acc.id_to_obj:
			continue
		obj4 = acc.id_to_obj[tag_id]                 # (4,3)
		# Canonicalize the detected quad to match the board's corner order
		c_ord = reorder_corners_to_board(c.copy(), obj4).reshape(-1, 2)
		obj_pts.append(obj4)
		img_pts.append(c_ord)
	if not obj_pts:
		return None, None
	obj = np.concatenate(obj_pts, axis=0).astype(np.float32).reshape(-1, 1, 3)
	img = np.concatenate(img_pts, axis=0).astype(np.float32).reshape(-1, 1, 2)
	return obj, img


def _compute_errors(obj, img, K, D):
	ok, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 3), img.reshape(-1, 2), K, D, flags=cv2.SOLVEPNP_ITERATIVE)
	if not ok:
		# try EPNP as a fallback
		ok, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 3), img.reshape(-1, 2), K, D, flags=cv2.SOLVEPNP_EPNP)
		if not ok:
			return None, None, None, None
	proj, _ = cv2.projectPoints(obj.reshape(-1, 3), rvec, tvec, K, D)
	proj = proj.reshape(-1, 2)
	img2 = img.reshape(-1, 2)
	err_vec = (proj - img2)  # (N,2) vector from detected -> reprojected
	err_mag = np.linalg.norm(err_vec, axis=1)  # (N,)
	return rvec, tvec, err_vec, err_mag


def _plot_quiver(img_bgr, img_points, err_vec, out_path, title):
	h, w = img_bgr.shape[:2]
	plt.figure(figsize=(w/200.0, h/200.0), dpi=200)
	plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
	U = err_vec[:, 0]
	V = err_vec[:, 1]
	X = img_points[:, 0]
	Y = img_points[:, 1]
	# scale arrows for visibility without overwhelming
	plt.quiver(X, Y, U, V, color="lime", angles="xy", scale_units="xy", scale=1.0, width=0.0025)
	plt.title(title)
	plt.gca().invert_yaxis()
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.savefig(out_path, dpi=200)
	plt.close()


def _plot_magnitude(img_bgr, img_points, err_mag, out_path, title):
	h, w = img_bgr.shape[:2]
	plt.figure(figsize=(w/200.0, h/200.0), dpi=200)
	plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
	sc = plt.scatter(img_points[:, 0], img_points[:, 1], c=err_mag, s=16, cmap="magma", alpha=0.9)
	cb = plt.colorbar(sc, fraction=0.046, pad=0.04)
	cb.set_label("Reprojection error (px)")
	plt.title(title)
	plt.gca().invert_yaxis()
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.savefig(out_path, dpi=200)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Plot reprojection error vectors and magnitudes from latest calibration JSON and a diagnostic image.")
	parser.add_argument("--json", type=str, default=None, help="Path to mono_calibration_latest.json")
	parser.add_argument("--image", type=str, default=None, help="Path to a diagnostic or sample image to analyze")
	parser.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to data/)")
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	repo_root = os.path.normpath(os.path.join(base_dir, ".."))
	data_dir = os.path.join(repo_root, "data")

	json_path = args.json or os.path.join(data_dir, "mono_calibration_latest.json")
	if not os.path.exists(json_path):
		print("[ERR] JSON not found:", json_path)
		return

	K, D, image_size, meta = _load_calibration(json_path)

	img_path = args.image or _auto_find_latest_diag(data_dir)
	if img_path is None or not os.path.exists(img_path):
		print("[ERR] No image provided and no *_calib_diag_*.png found in data/.")
		return

	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	if img is None:
		print("[ERR] Could not read image:", img_path)
		return

	board = _make_board()
	acc, corners, ids = _detect_from_image(img, board)
	if ids is None or len(ids) == 0:
		print("[ERR] No tags detected in image; cannot compute errors.")
		return

	obj, img_pts = _build_correspondences(acc, corners, ids)
	if obj is None:
		print("[ERR] Could not build correspondences.")
		return

	rvec, tvec, err_vec, err_mag = _compute_errors(obj, img_pts, K, D)
	if err_vec is None:
		print("[ERR] PnP failed; cannot compute errors.")
		return

	out_dir = args.out_dir or data_dir
	os.makedirs(out_dir, exist_ok=True)

	stamp = os.path.basename(img_path).replace(".png", "")
	title_base = f"Reprojection errors ({os.path.basename(json_path)})"

	q_out = os.path.join(out_dir, f"reproj_quiver_{stamp}.png")
	_plot_quiver(img, img_pts.reshape(-1, 2), err_vec, q_out, title_base + " - vectors")

	m_out = os.path.join(out_dir, f"reproj_magnitude_{stamp}.png")
	_plot_magnitude(img, img_pts.reshape(-1, 2), err_mag, m_out, title_base + " - magnitude")

	print(f"[SAVE] Wrote {q_out}")
	print(f"[SAVE] Wrote {m_out}")


if __name__ == "__main__":
	main()



