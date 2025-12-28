from pathlib import Path
from collections.abc import Mapping

import joblib
import numpy as np  # ensure numpy is available when unpickling


def inspect_file(pickle_path: str) -> None:
    p = Path(pickle_path)
    print(f"Loading: {p}")
    data = joblib.load(p)
    print("Top-level type:", type(data))

    if not isinstance(data, Mapping):
        print(data)
        return

    keys = sorted(data.keys())
    print("Num frames:", len(keys))
    print("First 5 frame keys:", keys[:5])

    # Inspect one representative frame
    first_key = keys[0]
    frame = data[first_key]
    print("\nFirst frame key:", first_key)
    print("Frame dict type:", type(frame))

    if isinstance(frame, Mapping):
        print("Frame keys:", list(frame.keys()))
        for k, v in frame.items():
            print(f"\n--- {k} ---")
            print("  type:", type(v))
            if hasattr(v, "shape"):
                print("  shape:", v.shape)
            elif hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                print("  len:", len(v))

        # Drill into a few important fields for the first tracked person (index 0)
        def show_inner(name: str):
            if name not in frame:
                return
            lst = frame[name]
            if not lst:
                return
            inner = lst[0]
            print(f"\n>>> {name}[0] details <<<")
            print("  type:", type(inner))
            if hasattr(inner, "shape"):
                print("  shape:", inner.shape)

        for key in ["3d_joints", "2d_joints", "pose", "smpl", "camera", "camera_bbox"]:
            show_inner(key)
    else:
        print(frame)


if __name__ == "__main__":
    # Update this path to point to the left/right 4D Humans result you want to inspect
    inspect_file(r"runs\my_swing_stageA_120front\less_clutter120cam1.pkl")