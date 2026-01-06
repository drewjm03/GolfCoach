"""
Thin wrapper to expose the no-overlap stereo calibrator under the expected module name.

This allows calling:
    python -m apps.calibration_testing.stereo_cam_calibrator_offline_no_overlap

while keeping the main implementation in stereo_calibrate_no_overlap.py.
"""

from . import stereo_calibrate_no_overlap as _impl


def main() -> None:
    _impl.main()


if __name__ == "__main__":
    main()


