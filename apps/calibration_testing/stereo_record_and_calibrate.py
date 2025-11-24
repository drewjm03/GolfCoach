"""
Stereo recording + offline calibration pipeline.

- Step 1: Runs the live stereo keyframe recorder to collect keyframes.
- Step 2: Runs the offline stereo calibrator on the recorded keyframes.
- Step 3: Prints the path to the produced calibration JSON for downstream triangulation.

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


def main():
	parser = argparse.ArgumentParser(description="Stereo pipeline: record keyframes then offline calibrate -> JSON for triangulation.")
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
	parser.add_argument("--corner-order", type=str, default=None,
	                    help="Corner order override 'i0,i1,i2,i3' (only applicable to grid8x5).")
	# Recording
	parser.add_argument("--target-keyframes", type=int, default=50, help="Number of keyframes to collect.")
	parser.add_argument("--accept-period", type=float, default=0.5, help="Minimum seconds between accepted keyframes.")
	parser.add_argument("--out-dir", type=str, default=None, help="Base output directory (default: repo_root/data).")
	# Calibrator diagnostics
	parser.add_argument("--save-all-diag", action="store_true", help="Save diagnostic overlays for all kept views.")

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

	# Echo summary for downstream tools
	try:
		with open(calib_json, "r", encoding="utf-8") as f:
			meta = json.load(f)
		image_size = tuple(meta.get("image_size", []))
		board_source = meta.get("board_source", "unknown")
		print(f"[PIPE] Calibration complete. JSON: {calib_json}")
		print(f"[PIPE] image_size={image_size} board_source={board_source}")
	except Exception:
		print(f"[PIPE] Calibration complete. JSON: {calib_json}")


if __name__ == "__main__":
	main()


