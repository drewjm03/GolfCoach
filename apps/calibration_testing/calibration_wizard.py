from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess


def _abs(p: str) -> str:
	return os.path.abspath(p)


def _run(cmd: list[str]) -> None:
	print("[WIZ] Running:", " ".join(cmd))
	subprocess.run(cmd, check=True)


def main() -> None:
	parser = argparse.ArgumentParser(description="Calibration Wizard: mono0 -> mono1 -> stereo extrinsics (no-overlap)")
	parser.add_argument("--cam0", type=int, required=True)
	parser.add_argument("--cam1", type=int, required=True)
	parser.add_argument("--board-source", type=str, choices=["harvard", "grid8x5"], required=True)

	parser.add_argument("--target-keyframes-mono", type=int, default=150)
	parser.add_argument("--target-keyframes-stereo", type=int, default=60)
	parser.add_argument("--accept-period", type=float, default=0.25)

	parser.add_argument("--out-root", type=str, default="runs")
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--rig-out", type=str, default="data/rig_config.json")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--no-prompts", action="store_true")

	# Board-related passthroughs
	parser.add_argument("--april-pickle", type=str, default=None)
	parser.add_argument("--harvard-tag-size-m", type=float, default=None)
	parser.add_argument("--harvard-tag-spacing-m", type=float, default=None)
	parser.add_argument("--corner-order", type=str, default=None)

	args = parser.parse_args()

	out_root = _abs(args.out_root)
	os.makedirs(out_root, exist_ok=True)

	run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
	run_dir = _abs(os.path.join(out_root, run_name))
	os.makedirs(run_dir, exist_ok=True)

	mono0_dir = _abs(os.path.join(run_dir, "mono_cam0_keyframes"))
	mono1_dir = _abs(os.path.join(run_dir, "mono_cam1_keyframes"))
	stereo_dir = _abs(os.path.join(run_dir, "stereo_extr_keyframes"))
	intr0_json = _abs(os.path.join(run_dir, "cam0_intrinsics.json"))
	intr1_json = _abs(os.path.join(run_dir, "cam1_intrinsics.json"))
	rig_out = _abs(args.rig_out)

	def maybe_prompt(msg: str) -> None:
		if not args.no_prompts:
			input(msg + " Press ENTER to continue...")

	# 1) Record MONO keyframes for cam0
	if args.resume and os.path.isdir(mono0_dir):
		print(f"[WIZ] Skipping mono cam0 record (exists): {mono0_dir}")
	else:
		maybe_prompt("[WIZ] Step 1 - Mono cam0. Place board in cam0 FoV. Move to cover corners/center.")
		cmd = [
			sys.executable, "-m", "apps.calibration_testing.mono_keyframe_recorder",
			"--board-source", str(args.board-source) if hasattr(args, "board-source") else str(args.board_source),
			"--cam", str(args.cam0),
			"--out-dir", mono0_dir,
			"--target-keyframes", str(args.target_keyframes_mono),
			"--accept-period", str(args.accept_period),
		]
		if args.april_pickle:
			cmd += ["--april-pickle", args.april_pickle]
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
		if args.corner_order:
			cmd += ["--corner-order", args.corner_order]
		_run(cmd)

	# 2) Record MONO keyframes for cam1
	if args.resume and os.path.isdir(mono1_dir):
		print(f"[WIZ] Skipping mono cam1 record (exists): {mono1_dir}")
	else:
		maybe_prompt("[WIZ] Step 2 - Mono cam1. Place board in cam1 FoV. Move to cover corners/center.")
		cmd = [
			sys.executable, "-m", "apps.calibration_testing.mono_keyframe_recorder",
			"--board-source", str(args.board_source),
			"--cam", str(args.cam1),
			"--out-dir", mono1_dir,
			"--target-keyframes", str(args.target_keyframes_mono),
			"--accept-period", str(args.accept_period),
		]
		if args.april_pickle:
			cmd += ["--april-pickle", args.april_pickle]
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
		if args.corner_order:
			cmd += ["--corner-order", args.corner_order]
		_run(cmd)

	# 3) Run mono calibration for cam0/cam1
	if args.resume and os.path.isfile(intr0_json):
		print(f"[WIZ] Skipping mono cam0 calib (exists): {intr0_json}")
	else:
		maybe_prompt("[WIZ] Step 3a - Mono calibration cam0.")
		cmd = [
			sys.executable, "-m", "apps.calibration_testing.mono_cam_calibrator_offline",
			"--keyframes-dir", mono0_dir,
			"--out-json", intr0_json,
			"--board-source", str(args.board_source),
		]
		if args.april_pickle:
			cmd += ["--april-pickle", args.april_pickle]
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
		if args.corner_order:
			cmd += ["--corner-order", args.corner_order]
		_run(cmd)

	if args.resume and os.path.isfile(intr1_json):
		print(f"[WIZ] Skipping mono cam1 calib (exists): {intr1_json}")
	else:
		maybe_prompt("[WIZ] Step 3b - Mono calibration cam1.")
		cmd = [
			sys.executable, "-m", "apps.calibration_testing.mono_cam_calibrator_offline",
			"--keyframes-dir", mono1_dir,
			"--out-json", intr1_json,
			"--board-source", str(args.board_source),
		]
		if args.april_pickle:
			cmd += ["--april-pickle", args.april_pickle]
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
		if args.corner_order:
			cmd += ["--corner-order", args.corner_order]
		_run(cmd)

	# 4) Record STEREO dataset (no-overlap)
	if args.resume and os.path.isdir(stereo_dir):
		print(f"[WIZ] Skipping stereo record (exists): {stereo_dir}")
	else:
		maybe_prompt("[WIZ] Step 4 - Stereo record (both cams see board simultaneously).")
		cmd = [
			sys.executable, "-m", "apps.calibration_testing.stereo_keyframe_recorder",
			"--board-source", str(args.board_source),
			"--cam0", str(args.cam0),
			"--cam1", str(args.cam1),
			"--target-keyframes", str(args.target_keyframes_stereo),
			"--accept-period", str(args.accept_period),
			"--out-dir", stereo_dir,
			"--no-overlap",
		]
		if args.april_pickle:
			cmd += ["--april-pickle", args.april_pickle]
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
		if args.corner_order:
			cmd += ["--corner-order", args.corner_order]
		_run(cmd)

	# 5) Solve stereo extrinsics (no-overlap PnP compose) with intrinsics fixed
	if args.resume and os.path.isfile(rig_out):
		print(f"[WIZ] Skipping final rig write (exists): {rig_out}")
	else:
		maybe_prompt("[WIZ] Step 5 - Stereo extrinsics from PnP compose with fixed intrinsics.")
		cmd = [
			sys.executable, "-m", "apps.calibration_testing.stereo_cam_calibrator_offline_no_overlap",
			"--keyframes-dir", stereo_dir,
			"--board-source", str(args.board_source),
			"--intrinsics0", intr0_json,
			"--intrinsics1", intr1_json,
			"--skip-mono",
			"--out-json", rig_out,
		]
		if args.april_pickle:
			cmd += ["--april-pickle", args.april_pickle]
		if args.harvard_tag_size_m is not None:
			cmd += ["--harvard-tag-size-m", str(args.harvard_tag_size_m)]
		if args.harvard_tag_spacing_m is not None:
			cmd += ["--harvard-tag-spacing-m", str(args.harvard_tag_spacing_m)]
		if args.corner_order:
			cmd += ["--corner-order", args.corner_order]
		_run(cmd)

	print("[WIZ] Done.")
	print(f"[WIZ] Outputs:")
	print(f"  - {intr0_json}")
	print(f"  - {intr1_json}")
	print(f"  - {rig_out}")


if __name__ == "__main__":
	main()


