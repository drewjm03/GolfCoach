import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def load_ts_json(path: Path) -> Tuple[np.ndarray, float]:
    """Load timestamps and written_fps from a stats JSON file."""
    with path.open("r") as f:
        data = json.load(f)

    if "ts" not in data:
        raise KeyError(f"JSON file {path} does not contain 'ts' array")

    ts = np.asarray(data["ts"], dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError(f"Timestamps in {path} must be a 1D array")

    # Ensure timestamps are non-decreasing
    if len(ts) > 1 and not np.all(np.diff(ts) >= 0):
        raise ValueError(f"Timestamps in {path} are not non-decreasing")

    written_fps = float(data.get("written_fps", 0.0) or 0.0)
    return ts, written_fps


def choose_master(
    left_ts: np.ndarray,
    right_ts: np.ndarray,
    requested: str,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Choose master stream.

    Returns:
        master_side: "left" or "right"
        master_ts
        other_ts
    """
    if requested == "auto":
        # Pick stream with fewer timestamps as master
        if len(left_ts) <= len(right_ts):
            master_side = "left"
        else:
            master_side = "right"
    elif requested in ("left", "right"):
        master_side = requested
    else:
        raise ValueError(f"Invalid master side: {requested}")

    if master_side == "left":
        return "left", left_ts, right_ts
    else:
        return "right", right_ts, left_ts


def estimate_offset_sec(
    master_ts: np.ndarray,
    other_ts: np.ndarray,
    max_samples: int = 200,
) -> float:
    """
    Coarse estimate of constant offset between streams.

    For each of the first N master timestamps, find nearest in other.
    Take median(other_ts[j] - master_ts[i]).
    """
    if master_ts.size == 0 or other_ts.size == 0:
        raise RuntimeError("Cannot estimate offset: one of the timestamp arrays is empty")

    n = min(max_samples, master_ts.size)
    offsets: List[float] = []
    for i in range(n):
        t = master_ts[i]
        k = int(np.searchsorted(other_ts, t, side="left"))
        candidates: List[int] = []
        if 0 <= k < other_ts.size:
            candidates.append(k)
        if 0 <= k - 1 < other_ts.size:
            candidates.append(k - 1)
        if not candidates:
            continue
        j = min(candidates, key=lambda idx: abs(other_ts[idx] - t))
        offsets.append(float(other_ts[j] - t))

    if not offsets:
        raise RuntimeError("Failed to estimate offset: no nearby timestamps found")

    return float(np.median(offsets))


def monotonic_match(
    master_ts: np.ndarray,
    other_ts: np.ndarray,
    offset_sec: float,
    max_dt_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monotonic nearest-neighbor matching with dropping of frames.

    For each master index i in order:
        target time t = master_ts[i] + offset_sec
        find nearest j in other_ts with j >= last_j
        accept if abs(other_ts[j] - t) <= max_dt_sec

    Returns:
        master_indices (int64)
        other_indices (int64)
        dt_error_sec (float64): other_ts[j] - (master_ts[i] + offset_sec)
    """
    master_indices: List[int] = []
    other_indices: List[int] = []
    dt_errors: List[float] = []

    last_j = 0
    n_other = other_ts.size

    for i, t_master in enumerate(master_ts):
        if last_j >= n_other:
            break

        t_target = t_master + offset_sec
        k = int(np.searchsorted(other_ts, t_target, side="left"))
        if k < last_j:
            k = last_j

        candidates: List[int] = []
        if last_j <= k < n_other:
            candidates.append(k)
        if last_j <= k - 1 < n_other:
            candidates.append(k - 1)

        if not candidates:
            break

        j = min(candidates, key=lambda idx: abs(other_ts[idx] - t_target))
        dt = float(other_ts[j] - t_target)

        if abs(dt) <= max_dt_sec:
            master_indices.append(i)
            other_indices.append(j)
            dt_errors.append(dt)
            last_j = j + 1
        else:
            # Drop this master frame
            continue

    return (
        np.asarray(master_indices, dtype=np.int64),
        np.asarray(other_indices, dtype=np.int64),
        np.asarray(dt_errors, dtype=np.float64),
    )


def determine_output_fps(
    master_side: str,
    left_mp4: Path,
    right_mp4: Path,
    left_written_fps: float,
    right_written_fps: float,
) -> float:
    """Determine output FPS based on master video, then JSON, then fallback."""
    master_mp4 = left_mp4 if master_side == "left" else right_mp4
    cap = cv2.VideoCapture(str(master_mp4))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if fps <= 1e-6 or not np.isfinite(fps):
        written_fps = left_written_fps if master_side == "left" else right_written_fps
        if written_fps > 1e-6 and np.isfinite(written_fps):
            fps = written_fps
        else:
            fps = 120.0

    return fps


def write_synced_video(
    in_path: Path,
    out_path: Path,
    keep_indices: np.ndarray,
    fps: float,
) -> int:
    """
    Write a synced video by selecting frames with indices in keep_indices.

    Returns:
        Number of frames written.
    """
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    if keep_indices.size == 0:
        return 0

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {in_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {out_path}")

    keep_indices_sorted = np.sort(keep_indices)
    p = 0
    current_keep = int(keep_indices_sorted[p])
    written = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == current_keep:
            out.write(frame)
            written += 1
            p += 1
            if p >= keep_indices_sorted.size:
                break
            current_keep = int(keep_indices_sorted[p])

        frame_idx += 1

    cap.release()
    out.release()

    return written


def compute_dt_stats(dt_error_sec: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for dt_error_sec."""
    if dt_error_sec.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95_abs": 0.0,
            "max_abs": 0.0,
        }

    dt_abs = np.abs(dt_error_sec)
    return {
        "mean": float(np.mean(dt_error_sec)),
        "median": float(np.median(dt_error_sec)),
        "std": float(np.std(dt_error_sec)),
        "p95_abs": float(np.percentile(dt_abs, 95)),
        "max_abs": float(np.max(dt_abs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple stereo sync tool.")
    parser.add_argument("--left_mp4", type=str, required=True)
    parser.add_argument("--right_mp4", type=str, required=True)
    parser.add_argument("--left_json", type=str, required=True)
    parser.add_argument("--right_json", type=str, required=True)
    parser.add_argument("--out_left", type=str, required=True)
    parser.add_argument("--out_right", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)

    parser.add_argument(
        "--master",
        type=str,
        default="auto",
        choices=["auto", "left", "right"],
        help='Master stream to drive matching (default: "auto").',
    )
    parser.add_argument(
        "--max_dt_sec",
        type=float,
        default=0.008,
        help="Maximum allowed time difference between matched frames (seconds).",
    )
    parser.add_argument(
        "--offset_sec",
        type=str,
        default="auto",
        help='Constant time offset to apply to master timestamps (seconds), or "auto".',
    )
    parser.add_argument(
        "--limit_pairs",
        type=int,
        default=None,
        help="Optional limit on number of matched pairs (for debugging).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    left_mp4 = Path(args.left_mp4)
    right_mp4 = Path(args.right_mp4)
    left_json_path = Path(args.left_json)
    right_json_path = Path(args.right_json)
    out_left = Path(args.out_left)
    out_right = Path(args.out_right)
    out_json = Path(args.out_json)

    # 1) Load timestamps
    left_ts, left_written_fps = load_ts_json(left_json_path)
    right_ts, right_written_fps = load_ts_json(right_json_path)

    # 2) Choose master stream
    master_side, master_ts, other_ts = choose_master(
        left_ts=left_ts,
        right_ts=right_ts,
        requested=args.master,
    )

    # 3) Estimate constant offset (or use user-provided)
    if isinstance(args.offset_sec, str) and args.offset_sec.lower() == "auto":
        offset_sec = estimate_offset_sec(master_ts, other_ts)
    else:
        try:
            offset_sec = float(args.offset_sec)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid offset_sec value: {args.offset_sec}") from exc

    # 4) Monotonic nearest-neighbor matching
    master_indices, other_indices, dt_error_sec = monotonic_match(
        master_ts=master_ts,
        other_ts=other_ts,
        offset_sec=offset_sec,
        max_dt_sec=float(args.max_dt_sec),
    )

    if args.limit_pairs is not None:
        limit = int(args.limit_pairs)
        master_indices = master_indices[:limit]
        other_indices = other_indices[:limit]
        dt_error_sec = dt_error_sec[:limit]

    matched_frames = int(master_indices.size)
    if matched_frames == 0:
        raise RuntimeError("No frame pairs matched. Try increasing --max_dt_sec or check timestamps.")

    # Map indices back to left/right
    if master_side == "left":
        left_indices = master_indices
        right_indices = other_indices
    else:
        left_indices = other_indices
        right_indices = master_indices

    dropped_master_frames = int(master_ts.size - matched_frames)
    dropped_other_frames = int(other_ts.size - matched_frames)

    # 5) Write synced videos
    # Ensure output directories exist
    out_left.parent.mkdir(parents=True, exist_ok=True)
    out_right.parent.mkdir(parents=True, exist_ok=True)

    output_fps = determine_output_fps(
        master_side=master_side,
        left_mp4=left_mp4,
        right_mp4=right_mp4,
        left_written_fps=left_written_fps,
        right_written_fps=right_written_fps,
    )

    written_left = write_synced_video(
        in_path=left_mp4,
        out_path=out_left,
        keep_indices=left_indices,
        fps=output_fps,
    )
    written_right = write_synced_video(
        in_path=right_mp4,
        out_path=out_right,
        keep_indices=right_indices,
        fps=output_fps,
    )

    if written_left != matched_frames or written_right != matched_frames:
        raise RuntimeError(
            f"Mismatch between matched frames ({matched_frames}) and written frames "
            f"(left={written_left}, right={written_right})."
        )

    # Prepare kept timestamps
    t_left_kept = left_ts[left_indices].astype(float).tolist()
    t_right_kept = right_ts[right_indices].astype(float).tolist()

    # 6) Save sync_mapping.json
    dt_stats = compute_dt_stats(dt_error_sec)

    sync_mapping: Dict[str, Any] = {
        "inputs": {
            "left_mp4": str(left_mp4),
            "right_mp4": str(right_mp4),
            "left_json": str(left_json_path),
            "right_json": str(right_json_path),
        },
        "settings": {
            "master": master_side,
            "max_dt_sec": float(args.max_dt_sec),
            "offset_sec": float(offset_sec),
        },
        "results": {
            "matched_frames": matched_frames,
            "left_indices": left_indices.astype(int).tolist(),
            "right_indices": right_indices.astype(int).tolist(),
            "dt_error_sec": dt_error_sec.astype(float).tolist(),
            "dt_error_stats": dt_stats,
            "dropped_master_frames": dropped_master_frames,
            "dropped_other_frames": dropped_other_frames,
            "t_left_kept": t_left_kept,
            "t_right_kept": t_right_kept,
        },
    }

    with out_json.open("w") as f:
        json.dump(sync_mapping, f, indent=2)

    # 7) Print summary
    dt_abs = np.abs(dt_error_sec)
    median_abs = float(np.median(dt_abs)) if dt_abs.size > 0 else 0.0
    p95_abs = float(np.percentile(dt_abs, 95)) if dt_abs.size > 0 else 0.0

    print(f"Master side: {master_side}")
    print(f"Offset (sec): {offset_sec:.6f}")
    print(f"Matched frames: {matched_frames}")
    print(f"Median |dt_error_sec| (ms): {median_abs * 1000.0:.3f}")
    print(f"p95 |dt_error_sec| (ms): {p95_abs * 1000.0:.3f}")
    print(f"Dropped master frames: {dropped_master_frames}")
    print(f"Dropped other frames: {dropped_other_frames}")


if __name__ == "__main__":
    main()


