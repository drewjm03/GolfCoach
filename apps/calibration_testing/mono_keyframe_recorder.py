"""
Live mono keyframe recorder.

- Derived from stereo_keyframe_recorder.py, simplified to a single camera.
- Saves keyframes in a simple structure:
  - frame_XXX.png
  - frame_XXX.json with ids/corners and image_size
  - meta.json describing board/recording config
"""

import os
import sys
import time
import json
import argparse
import queue
from typing import Optional

import numpy as np
import cv2

# ---- local imports ----
try:
	from .. import config
	from ..capture import CamReader
	from ..detect import CalibrationAccumulator
	from ..stereo_calib_plot import load_board
except Exception:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from apps import config  # type: ignore
	from apps.capture import CamReader  # type: ignore
	from apps.detect import CalibrationAccumulator  # type: ignore
	from apps.stereo_calib_plot import load_board  # type: ignore


def _ensure_out_dir(path: str) -> str:
	path = os.path.abspath(path)
	os.makedirs(path, exist_ok=True)
	return path


def main() -> None:
	parser = argparse.ArgumentParser(description="Live mono keyframe recorder (grid/Harvard).")
	parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], required=True)
	parser.add_argument("--april-pickle", type=str, default=None,
	                    help="Path to local AprilBoards.pickle (for Harvard board).")
	parser.add_argument("--harvard-tag-size-m", type=float, default=None,
	                    help="Tag side length in meters for Harvard board.")
	parser.add_argument("--harvard-tag-spacing-m", type=float, default=None,
	                    help="Tag spacing (meters) for Harvard board (informational).")
	parser.add_argument("--target-keyframes", type=int, default=150,
	                    help="Number of mono keyframes to record before exiting.")
	parser.add_argument("--accept-period", type=float, default=0.25,
	                    help="Minimum seconds between accepted keyframes.")
	parser.add_argument("--out-dir", type=str, required=True,
	                    help="Output directory to write keyframes/meta.")
	parser.add_argument("--cam", type=int, required=True, help="Camera index.")
	parser.add_argument("--corner-order", type=str, default=None,
	                    help="Manual corner order override as four comma-separated indices, e.g. '0,1,2,3'.")

	args, _ = parser.parse_known_args()

	# Open camera
	print(f"[MONO-REC] Opening camera cam={args.cam}")
	cam = CamReader(int(args.cam))
	ts, frame0 = cam.latest()
	H, W = frame0.shape[:2]
	image_size = (W, H)

	# Build board and accumulator
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
				print(f"[APRIL] Using manual corner order: {corner_order_override}")
			else:
				print(f"[WARN] Ignoring invalid --corner-order '{args.corner_order}'. Expected 4 comma-separated indices 0..3.")
		except Exception as e:
			print(f"[WARN] Failed to parse --corner-order '{args.corner_order}': {e}. Ignoring.")
	else:
		# For Harvard, mirror mono default used elsewhere
		if str(args.board_source).lower().strip() == "harvard":
			corner_order_override = [3, 0, 1, 2]
			disable_autoreorder = True
			print("[APRIL] Using default per-tag corner order 3,0,1,2 for Harvard board")

	acc = CalibrationAccumulator(
		board,
		image_size,
		corner_order_override=corner_order_override,
		disable_corner_autoreorder=disable_autoreorder,
	)
	print("[APRIL] Backend:", acc.get_backend_name())
	print("[APRIL] Families:", acc._apriltag_family_string())

	# Output directory
	record_dir = _ensure_out_dir(args.out_dir)

	meta = {
		"mode": "mono",
		"cam": int(args.cam),
		"image_size": [int(W), int(H)],
		"APRIL_DICT": int(getattr(config, "APRIL_DICT", 0)),
		"board_source": str(args.board_source),
		"april_pickle": args.april_pickle or "",
		"harvard_tag_size_m": float(args.harvard_tag_size_m) if args.harvard_tag_size_m is not None else None,
		"harvard_tag_spacing_m": float(args.harvard_tag_spacing_m) if args.harvard_tag_spacing_m is not None else None,
		"corner_order": args.corner_order or "",
		"target_keyframes": int(args.target_keyframes),
		"accept_period": float(args.accept_period),
		"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
	}
	with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)
	print(f"[MONO-REC] Recording mono keyframes to {record_dir}")

	# UI
	win = "Mono Keyframe Recorder"
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)

	keyframes = 0
	last_accept = 0.0

	try:
		while keyframes < args.target_keyframes:
			try:
				ts, f = cam.latest()
			except queue.Empty:
				time.sleep(0.01)
				continue

			g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

			# Detect tags
			try:
				corners, ids = acc.detect(g)
			except Exception:
				corners, ids = [], None

			n = 0 if ids is None else len(ids)

			now = time.perf_counter()
			added = False
			if (now - last_accept) >= float(args.accept_period):
				ok = (corners is not None and ids is not None and len(corners) >= config.MIN_MARKERS_PER_VIEW)
				if ok:
					# Save keyframe artifacts
					idx = keyframes
					out_img = os.path.join(record_dir, f"frame_{idx:03d}.png")
					cv2.imwrite(out_img, f)

					out_js = os.path.join(record_dir, f"frame_{idx:03d}.json")
					ids_list = [] if ids is None else [int(iv[0]) for iv in ids]
					corners_list = [] if not corners else [c.reshape(4, 2).tolist() for c in corners]
					with open(out_js, "w", encoding="utf-8") as jf:
						json.dump(
							{
								"frame_idx": int(idx),
								"t_sec": float(ts) if isinstance(ts, (int, float)) else 0.0,
								"ids": ids_list,
								"corners": corners_list,
								"image_size": [int(W), int(H)],
							},
							jf,
							indent=2,
						)
					last_accept = now
					keyframes += 1
					added = True
					print(f"[MONO-REC] Keyframes: {keyframes}  (tags={n})")

			# Annotate and display
			vis = f.copy()
			try:
				if corners is not None and ids is not None and len(corners) > 0:
					cv2.aruco.drawDetectedMarkers(vis, corners, ids)
			except Exception:
				pass

			status_line = f"KF={keyframes}/{args.target_keyframes} tags={n}"
			cv2.putText(
				vis,
				status_line,
				(12, 28),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 255),
				2,
				lineType=cv2.LINE_AA,
			)
			cv2.imshow(win, vis)

			k = cv2.waitKey(1) & 0xFF
			if k in (27, ord("q")):
				print("[KEY] Quit.")
				break
	finally:
		cam.release()
		cv2.destroyAllWindows()
		print("[MONO-REC] Closed.")


if __name__ == "__main__":
	main()


