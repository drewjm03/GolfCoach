import os, sys, time, json, queue, cv2, argparse
import numpy as np

# Allow running as a module: python -m apps.calibration_testing.single_cam_calibrator
try:
	from .. import config
	from ..capture import CamReader, set_manual_exposure_uvc, set_auto_exposure_uvc, set_uvc_gain
	from ..detect import CalibrationAccumulator, board_ids_safe
	from ..sdk import try_load_dll, sdk_config
except Exception:
	# Fallback for direct execution if relative import fails
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from apps import config
	from apps.capture import CamReader, set_manual_exposure_uvc, set_auto_exposure_uvc, set_uvc_gain
	from apps.detect import CalibrationAccumulator, board_ids_safe
	from apps.sdk import try_load_dll, sdk_config


try:
	import winsound
	def beep_ok():
		try:
			winsound.Beep(1000, 180)
		except Exception:
			pass
except Exception:
	def beep_ok():
		pass


def _make_board():
	dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
	ids_grid = np.arange(config.TAGS_X * config.TAGS_Y, dtype=np.int32).reshape(config.TAGS_Y, config.TAGS_X)
	ids_grid = np.flipud(ids_grid)
	ids = ids_grid.reshape(-1, 1).astype(np.int32)
	board = cv2.aruco.GridBoard((config.TAGS_X, config.TAGS_Y), config.TAG_SIZE_M, config.TAG_SEP_M, dictionary, ids)
	return board


def _draw_status(img, lines, y0=30, dy=26, color=(0, 255, 255)):
	for k, line in enumerate(lines):
		cv2.putText(img, line, (16, y0 + dy * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def _compute_detection_metrics(corners, ids):
	centers = []
	side_lengths = []
	for c in corners:
		c4 = c.reshape(4, 2).astype(np.float32)
		centers.append(c4.mean(axis=0))
		side = float(sum(np.linalg.norm(c4[(i + 1) % 4] - c4[i]) for i in range(4)) / 4.0)
		side_lengths.append(side)
	if not centers:
		return None
	centers = np.stack(centers, axis=0)
	global_center = centers.mean(axis=0)
	mean_side = float(np.mean(side_lengths)) if side_lengths else 0.0
	return global_center, mean_side


def _choose_best_keyframe(keyframes):
	if not keyframes:
		return None
	# keyframes entries: (gray, corners, ids, frame_bgr)
	return max(keyframes, key=lambda t: (0 if t[2] is None else len(t[2])))


def _best_view_index(acc):
	best_i, best_n = -1, -1
	for i, ids_img in enumerate(acc.ids0):
		n = 0 if ids_img is None else len(ids_img)
		if n > best_n:
			best_n, best_i = n, i
	return best_i


def _has_coverage(corners, W, H, min_span=0.35):
	try:
		xs = [p[0] for ci in corners for p in ci.reshape(4, 2)]
		ys = [p[1] for ci in corners for p in ci.reshape(4, 2)]
		span_x = (max(xs) - min(xs)) / float(max(1, W))
		span_y = (max(ys) - min(ys)) / float(max(1, H))
		return max(span_x, span_y) >= float(min_span)
	except Exception:
		return False


def _save_diagnostic_image(path_png, frame_bgr, corners, ids, acc, K, D, rvec=None, tvec=None, use_fisheye=False):
	# Assemble object and image points (per single view)
	obj_pts = []
	img_pts = []
	for c, iv in zip(corners, ids):
		tag_id = int(iv[0])
		if tag_id not in acc.id_to_obj:
			continue
		obj = acc.id_to_obj[tag_id]          # 4x3
		img = c.reshape(-1, 2).astype(np.float32)  # 4x2
		obj_pts.append(obj)
		img_pts.append(img)
	if not obj_pts:
		cv2.imwrite(path_png, frame_bgr)
		return
	obj_pts = np.concatenate(obj_pts, axis=0).astype(np.float32)
	img_pts = np.concatenate(img_pts, axis=0).astype(np.float32)

	# Get extrinsics for this view
	if rvec is None or tvec is None:
		ok, rvec_est, tvec_est = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
		if not ok:
			cv2.imwrite(path_png, frame_bgr)
			return
		rvec, tvec = rvec_est, tvec_est

	# Build mapping per-tag for center comparison
	# We will re-project per-tag 4 corners
	img = frame_bgr.copy()
	for c, iv in zip(corners, ids):
		tag_id = int(iv[0])
		if tag_id not in acc.id_to_obj:
			continue
		obj4 = acc.id_to_obj[tag_id].astype(np.float32)  # 4x3
		if use_fisheye:
			proj, _ = cv2.fisheye.projectPoints(obj4.reshape(1, -1, 3).astype(np.float64), rvec, tvec, K, D)  # 1x4x2
			proj = proj.reshape(-1, 2)
		else:
			proj, _ = cv2.projectPoints(obj4, rvec, tvec, K, D)  # 4x1x2
			proj = proj.reshape(-1, 2)
		det_center = c.reshape(4, 2).astype(np.float32).mean(axis=0)
		proj_center = proj.mean(axis=0)
		# draw detected center (cyan) and projected center (magenta)
		cv2.circle(img, (int(round(det_center[0])), int(round(det_center[1]))), 4, (255, 255, 0), -1)
		cv2.circle(img, (int(round(proj_center[0])), int(round(proj_center[1]))), 4, (255, 0, 255), -1)
		cv2.line(img,
		         (int(round(det_center[0])), int(round(det_center[1]))),
		         (int(round(proj_center[0])), int(round(proj_center[1]))),
		         (0, 180, 255), 1)

	cv2.putText(img, "Cyan=detected centers, Magenta=reprojected centers",
	            (16, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
	cv2.imwrite(path_png, img)


def main():
	auto_exposure_on = False
	current_exposure_step = config.DEFAULT_EXPOSURE_STEP
	current_gain = None

	# CLI: --cam-index / -i to select camera
	def _resolve_camera_index():
		parser = argparse.ArgumentParser(add_help=False)
		parser.add_argument("--cam-index", "-i", type=int, default=None, help="Camera index for single-camera calibration")
		args, _ = parser.parse_known_args()
		if args.cam_index is not None:
			return int(args.cam_index)
		env_order = os.environ.get("CAM_INDEX_ORDER", "").strip()
		if env_order:
			try:
				indices = [int(x.strip()) for x in env_order.split(",") if x.strip() != ""]
			except Exception:
				indices = [0]
			if not indices:
				indices = [0]
			return indices[0]
		return 0

	# Optional SDK pre-config
	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	dll, has_af, cam_ids = try_load_dll(base_dir)
	if cam_ids:
		for iid in cam_ids[:1]:
			sdk_config(dll, has_af, iid, fps=config.CAPTURE_FPS, lock_autos=True, anti_flicker_60hz=True,
			           exposure_us=(8000 if config.USE_SDK_EXPOSURE else None))
		time.sleep(0.3)

	print("[MAIN] Opening one camera…")
	index = _resolve_camera_index()
	print(f"[MAIN] Using camera index {index} (override via --cam-index or CAM_INDEX_ORDER)")

	cam = CamReader(index)
	try:
		g = cam.cap.get(cv2.CAP_PROP_GAIN)
		current_gain = float(g) if g is not None else None
		if current_gain is not None:
			print(f"[CV] Initial gain read-back: {current_gain}")
	except Exception:
		current_gain = None

	# One frame for size
	_, frame0 = cam.latest()
	H, W = frame0.shape[0], frame0.shape[1]
	image_size = (W, H)

	# Board and accumulator
	board = _make_board()
	acc = CalibrationAccumulator(board, image_size)
	print("[APRIL] Backend:", acc.get_backend_name())
	print("[APRIL] Families:", acc._apriltag_family_string())

	# Calibration state
	calibrating = False
	window_deadline = 0.0
	last_sample_t = 0.0
	keyframes = []  # list of (gray, corners, ids, frame_bgr)
	last_metrics = None  # (center, mean_side)
	last_accept_t = 0.0
	MIN_SAMPLE_PERIOD_S = 0.5
	CENTER_DELTA_PX = 30.0
	SCALE_DELTA_RATIO = 0.10

	win = "Single Cam Calibrator"
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)
	try:
		cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
	except Exception:
		pass

	try:
		while True:
			try:
				ts, frame = cam.latest()
			except queue.Empty:
				time.sleep(0.01)
				continue

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			now = time.perf_counter()

			annotated = frame.copy()

			if calibrating and now >= window_deadline:
				# End calibration window and run mono calibration on collected keyframes
				calibrating = False
				beep_ok()
				print(f"[CAL] Window complete. Collected {len(keyframes)} keyframes.")
				# Build per-image obj/img points from accumulated samples
				def _build_points_lists():
					obj_list, img_list = [], []
					for corners_img, ids_img in zip(acc.corners0, acc.ids0):
						obj_pts = []
						img_pts = []
						for c, idv in zip(corners_img, ids_img):
							tag_id = int(idv[0])
							if tag_id not in acc.id_to_obj:
								continue
							obj_pts.append(acc.id_to_obj[tag_id])  # 4x3
							img_pts.append(c.reshape(-1, 2))       # keep detected order
						if obj_pts:
							obj = np.concatenate(obj_pts, axis=0).astype(np.float64).reshape(-1, 1, 3)
							img = np.concatenate(img_pts, axis=0).astype(np.float64).reshape(-1, 1, 2)
							obj_list.append(obj)
							img_list.append(img)
					return obj_list, img_list

				obj_list, img_list = _build_points_lists()
				if not obj_list:
					print("[CAL] Not enough valid samples to calibrate.")
				else:
					used_fisheye = False
					K = np.eye(3, dtype=np.float64)
					D = np.zeros((4, 1), dtype=np.float64)
					rvecs = []; tvecs = []
					try:
						flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
						         cv2.fisheye.CALIB_CHECK_COND |
						         cv2.fisheye.CALIB_FIX_SKEW)
						criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
						rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
							obj_list, img_list, image_size, K, D, None, None,
							flags=flags, criteria=criteria)
						used_fisheye = True
						print(f"[CAL] Fisheye RMS: {float(rms):.4f}")
					except Exception as e:
						print("[CAL] Fisheye calibration failed:", e)
						# Fallback to standard model to get a good seed
						try:
							obj_std = [o.reshape(-1, 3).astype(np.float32) for o in obj_list]
							img_std = [i.reshape(-1, 2).astype(np.float32) for i in img_list]
							Ks = np.eye(3, dtype=np.float64)
							Ds = np.zeros((5, 1), dtype=np.float64)
							rms_mono, Ks, Ds, rvecs_mono, tvecs_mono = cv2.calibrateCamera(obj_std, img_std, image_size, Ks, Ds)
							print(f"[CAL] Mono RMS: {float(rms_mono):.4f}")

							# Prefilter views via quick pinhole PnP reprojection check
							def _prefilter_views(objL, imgL, Kp, Dp, thr_px=5.0):
								keep = []
								for vi, (o, im) in enumerate(zip(objL, imgL)):
                                    # o: (N,1,3), im: (N,1,2)
									o_std = o.reshape(-1, 3).astype(np.float32)
									im_std = im.reshape(-1, 2).astype(np.float32)
									ok_pnp, rv, tv = cv2.solvePnP(o_std, im_std, Kp, Dp, flags=cv2.SOLVEPNP_EPNP)
									if not ok_pnp:
										continue
									proj, _ = cv2.projectPoints(o_std, rv, tv, Kp, Dp)
									err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - im_std, axis=1)))
									if err < float(thr_px):
										keep.append(vi)
								return [objL[i] for i in keep], [imgL[i] for i in keep], keep

							obj_list_f, img_list_f, keep_idx = _prefilter_views(obj_list, img_list, Ks, Ds)
							if len(obj_list_f) >= 3:
								# Progressive fisheye with intrinsic guess
								K_guess = Ks.copy().astype(np.float64)
								D_guess = np.zeros((4, 1), dtype=np.float64)
								flags_fe = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
								            cv2.fisheye.CALIB_CHECK_COND |
								            cv2.fisheye.CALIB_FIX_SKEW |
								            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
								            cv2.fisheye.CALIB_FIX_K3 |
								            cv2.fisheye.CALIB_FIX_K4)
								rms1, K1, D1, rvecs1, tvecs1 = cv2.fisheye.calibrate(
									obj_list_f, img_list_f, image_size, K_guess, D_guess, None, None,
									flags=flags_fe, criteria=criteria)
								# release k3
								flags_fe &= ~cv2.fisheye.CALIB_FIX_K3
								rms2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
									obj_list_f, img_list_f, image_size, K1, D1, None, None,
									flags=flags_fe, criteria=criteria)
								# release k4
								flags_fe &= ~cv2.fisheye.CALIB_FIX_K4
								rms3, K3, D3, rvecs3, tvecs3 = cv2.fisheye.calibrate(
									obj_list_f, img_list_f, image_size, K2, D2, None, None,
									flags=flags_fe, criteria=criteria)
								used_fisheye = True
								K, D = K3, D3
								# rvecs/tvecs correspond to filtered views
								rvecs, tvecs = rvecs3, tvecs3
								print(f"[CAL] Fisheye (progressive) RMS: {float(rms3):.4f}")
							else:
								# Keep mono result
								K, D, rvecs, tvecs = Ks, Ds, rvecs_mono, tvecs_mono
						except Exception as ee:
							print("[CAL] Calibration failed:", ee)
							rms, K, D = None, None, None

					if K is not None:
						# Save diagnostic image using matched rvec/tvec from the best keyframe
						# Choose best view by id count, then align to calibration's rvecs/tvecs order
						i_best_global = _best_view_index(acc)
						if i_best_global < 0 or i_best_global >= len(keyframes):
							i_best_global = min(len(keyframes) - 1, 0)
							_, best_corners, best_ids, best_bgr = keyframes[i_best_global]
							repo_root = os.path.normpath(os.path.join(base_dir, ".."))
							out_dir = os.path.join(repo_root, "data")
							os.makedirs(out_dir, exist_ok=True)
							stamp = time.strftime("%Y%m%d_%H%M%S")
							prefix = "fisheye" if used_fisheye else "mono"
							out_png = os.path.join(out_dir, f"{prefix}_calib_diag_{stamp}.png")

							# Map to calibration view index; if we filtered fisheye views, map global->filtered
							rv, tv = None, None
							try:
								if used_fisheye and 'keep_idx' in locals() and isinstance(keep_idx, list) and keep_idx:
									if i_best_global in keep_idx:
										j = keep_idx.index(i_best_global)
									else:
										j = 0
									rv = rvecs[j] if j < len(rvecs) else None
									tv = tvecs[j] if j < len(tvecs) else None
								else:
									# Standard mono or unfiltered fisheye
									rv = rvecs[i_best_global] if isinstance(rvecs, (list, tuple)) and i_best_global < len(rvecs) else None
									tv = tvecs[i_best_global] if isinstance(tvecs, (list, tuple)) and i_best_global < len(tvecs) else None
							except Exception:
								rv, tv = None, None

							_save_diagnostic_image(out_png, best_bgr, best_corners, best_ids, acc, K, D,
							                       rvec=rv, tvec=tv, use_fisheye=used_fisheye)
							print(f"[SAVE] Diagnostic image -> {out_png}")
						# Persist intrinsics
						try:
							out_json = os.path.join(os.path.normpath(os.path.join(base_dir, "..")), "data", "mono_calibration_latest.json")
							os.makedirs(os.path.dirname(out_json), exist_ok=True)
							payload = {
								"image_size": list(image_size),
								"K": K.tolist(),
								"D": D.tolist(),
								"rms": (float(rms) if rms is not None else None),
								"num_keyframes": int(len(keyframes)),
								"model": ("fisheye" if used_fisheye else "standard"),
							}
							with open(out_json, "w", encoding="utf-8") as f:
								json.dump(payload, f, indent=2)
							print(f"[SAVE] Wrote {out_json}")
						except Exception as e:
							print("[SAVE] JSON write failed:", e)

					# Reset for next run
					keyframes.clear()
					last_metrics = None
					last_accept_t = 0.0

			# When calibrating, gate samples to "keyframes"
			if calibrating and (now - last_sample_t) >= MIN_SAMPLE_PERIOD_S:
				last_sample_t = now
				try:
					corners, ids = acc.detect(gray)
				except Exception:
					corners, ids = [], None
				n_ids = 0 if ids is None else len(ids)
				if n_ids >= config.MIN_MARKERS_PER_VIEW and corners and _has_coverage(corners, W, H, min_span=0.30):
					metrics = _compute_detection_metrics(corners, ids)
					accept = False
					if metrics is None:
						accept = False
					else:
						center, mean_side = metrics
						if last_metrics is None:
							accept = True
						else:
							last_center, last_side = last_metrics
							center_delta = float(np.linalg.norm(center - last_center))
							scale_delta = abs(mean_side - last_side) / max(1e-3, last_side)
							if center_delta >= CENTER_DELTA_PX or scale_delta >= SCALE_DELTA_RATIO:
								accept = True
							elif (now - last_accept_t) >= max(config.CALIB_SAMPLE_PERIOD_S, 3.0):
								# Fallback: accept if some time has passed to add diversity
								accept = True
						if accept:
							ok = acc._accumulate_single(0, corners, ids)
							if ok:
								keyframes.append((gray.copy(), [c.copy() for c in corners], ids.copy(), frame.copy()))
								last_metrics = (center, mean_side)
								last_accept_t = now
								print(f"[CAL] Keyframes: {len(keyframes)} (ids={n_ids})")

					# Draw detections
					try:
						cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
						for c, iv in zip(corners, ids if ids is not None else []):
							c4 = c.reshape(4, 2).astype(int)
							p = c4.mean(axis=0).astype(int)
							cv2.putText(annotated, str(int(iv[0])), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
					except Exception:
						pass

			# Status overlay
			status = []
			if calibrating:
				remain = max(0.0, window_deadline - now)
				status.append(f"Calibrating… keyframes={len(keyframes)}  remain={int(remain)}s")
			status.append(f"Detector: {acc.get_backend_name()}")
			if auto_exposure_on:
				status.append("Exposure: Auto (UVC)")
			else:
				status.append(f"Exposure: Manual step {int(current_exposure_step)}")
			if current_gain is not None:
				status.append(f"Gain: {current_gain:.1f}")
			_draw_status(annotated, status, y0=30)

			# Show preview
			cv2.imshow(win, annotated)
			key = cv2.waitKey(1) & 0xFF
			if key == 27 or key == ord('q'):
				break
			elif key == ord('c'):
				calibrating = not calibrating
				print(f"[KEY] Calibrating -> {calibrating}")
				if calibrating:
					# Reset window and buffers
					window_deadline = time.perf_counter() + 60.0
					keyframes.clear()
					acc.corners0.clear(); acc.ids0.clear(); acc.counter0.clear()
					last_sample_t = 0.0
					last_metrics = None
					last_accept_t = 0.0
			elif key == ord('e'):
				auto_exposure_on = not auto_exposure_on
				print(f"[KEY] Auto exposure -> {auto_exposure_on}")
				if auto_exposure_on:
					set_auto_exposure_uvc(cam.cap)
				else:
					set_manual_exposure_uvc(cam.cap, step=current_exposure_step)
			elif key == ord(',') or key == 44:
				if not auto_exposure_on:
					current_exposure_step = max(config.MIN_EXPOSURE_STEP, int(current_exposure_step) - 1)
					set_manual_exposure_uvc(cam.cap, step=current_exposure_step)
					print(f"[KEY] Exposure step -> {current_exposure_step}")
			elif key == ord('.') or key == 46:
				if not auto_exposure_on:
					current_exposure_step = min(config.MAX_EXPOSURE_STEP, int(current_exposure_step) + 1)
					set_manual_exposure_uvc(cam.cap, step=current_exposure_step)
					print(f"[KEY] Exposure step -> {current_exposure_step}")
			elif key == ord(';') or key == 59:
				try:
					if current_gain is None:
						current_gain = float(cam.cap.get(cv2.CAP_PROP_GAIN))
					current_gain = max(config.MIN_GAIN, current_gain - config.GAIN_DELTA)
					set_uvc_gain(cam.cap, current_gain)
					print(f"[KEY] Gain -> {current_gain}")
				except Exception as e:
					print("[KEY] Gain decrease failed:", e)
			elif key == ord("'") or key == 39:
				try:
					if current_gain is None:
						current_gain = float(cam.cap.get(cv2.CAP_PROP_GAIN))
					current_gain = min(config.MAX_GAIN, current_gain + config.GAIN_DELTA)
					set_uvc_gain(cam.cap, current_gain)
					print(f"[KEY] Gain -> {current_gain}")
				except Exception as e:
					print("[KEY] Gain increase failed:", e)
	finally:
		cam.release()
		cv2.destroyAllWindows()
		print("[MAIN] Closed")


if __name__ == "__main__":
	main()


