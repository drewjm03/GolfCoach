import os
import numpy as np
import cv2


def harvard_tag_size_from_data_or_env(data, cli_size=None):
    """Return tag size (meters) from CLI/env or from pickle metadata if present."""
    # CLI override
    if cli_size is not None:
        try:
            return float(cli_size)
        except Exception:
            pass
    # ENV override
    env = os.environ.get("HARVARD_TAG_SIZE_M", "").strip()
    if env:
        try:
            return float(env)
        except Exception:
            pass
    # Try from pickle metadata
    if isinstance(data, dict):
        for k in ("tag_size", "tag_side", "tag_width", "april_tag_side", "tag_length", "marker_length"):
            v = data.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        for pk in ("params", "metadata", "board_params"):
            sub = data.get(pk)
            if isinstance(sub, dict):
                for k in ("tag_size", "tag_side", "tag_width", "april_tag_side", "tag_length", "marker_length"):
                    v = sub.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
    return None


def parse_board_pickle(dictionary, data, tag_size_m=None, harvard_tag_spacing_m=None):
    """
    Parse a Harvard-style pickle into a cv2.aruco.Board, matching the mono script logic:
    - If centers are provided, only infer center scale when BOTH tag size and spacing are provided.
    - Require tag_size_m if only centers exist (no corners).
    - Construct corners CCW around center in the board plane.
    """
    try:
        # If data itself is already a Board
        if hasattr(data, "getObjPoints") and (hasattr(data, "ids") or hasattr(data, "getIds")):
            return data

        # Prefer known keys; otherwise scan for plausible content
        candidate_keys = ["at_board_d", "at_coarseboard", "board", "at_board"]
        board_data = None
        for k in candidate_keys:
            if k in getattr(data, "keys", lambda: [])():
                board_data = data.get(k)
                break
        if board_data is None and isinstance(data, dict):
            for k, v in data.items():
                if hasattr(v, "getObjPoints") and (hasattr(v, "ids") or hasattr(v, "getIds")):
                    board_data = v; break
            if board_data is None and any(isinstance(v, (dict, list, tuple)) for v in data.values()):
                for k, v in data.items():
                    if isinstance(v, dict) and ("ids" in v or "objPoints" in v or "corners" in v):
                        board_data = v; break
                    if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
                        board_data = v; break
        if board_data is None:
            print(f"[BOARD] Available pickle keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            return None

        # If board_data is already a Board
        if hasattr(board_data, "getObjPoints") and (hasattr(board_data, "ids") or hasattr(board_data, "getIds")):
            return board_data

        obj_points = []
        ids_list = []
        if isinstance(board_data, dict) and 'ids' in board_data:
            ids_raw = list(board_data.get('ids', []))
            corners_raw = board_data.get('objPoints', board_data.get('corners', []))
            for tid, pts in zip(ids_raw, corners_raw):
                P = np.array(pts, dtype=np.float32).reshape(-1, 3 if np.array(pts).shape[-1] == 3 else 2)
                if P.shape[1] == 2:
                    P = np.hstack([P, np.zeros((P.shape[0], 1), dtype=np.float32)])
                if P.shape[0] == 4:
                    obj_points.append(P.reshape(4, 3))
                    ids_list.append(int(tid))
        elif isinstance(board_data, (list, tuple)) and len(board_data) > 0:
            # Collect centers to optionally infer unit->meter scale from tag size + spacing
            centers_units = []
            for item in board_data:
                if isinstance(item, dict) and 'tag_id' in item and 'center' in item:
                    c = np.array(item['center'], dtype=np.float32).reshape(-1)
                    centers_units.append(c[:2].astype(np.float32))
            center_scale = None
            if centers_units and (tag_size_m is not None) and (harvard_tag_spacing_m is not None):
                desired_cc_m = float(tag_size_m) + float(harvard_tag_spacing_m)
                if desired_cc_m > 0:
                    C = np.vstack(centers_units)
                    dists = []
                    for i in range(C.shape[0]):
                        di = np.hypot(C[i,0]-C[:,0], C[i,1]-C[:,1])
                        di = di[di > 1e-6]
                        if di.size > 0:
                            dists.append(np.min(di))
                    if dists:
                        nn_units = float(np.median(np.array(dists, dtype=np.float32)))
                        if nn_units > 0:
                            center_scale = desired_cc_m / nn_units
                            print(f"[BOARD] Harvard center scale inferred: {center_scale:.6f} m/unit (nn={nn_units:.6f} units, cc={desired_cc_m:.6f} m)")
            for item in board_data:
                if not isinstance(item, dict):
                    continue
                # Case A: explicit corners
                if 'id' in item and ('corners' in item or 'objPoints' in item):
                    tid = int(item['id'])
                    pts = item.get('objPoints', item.get('corners'))
                    P = np.array(pts, dtype=np.float32).reshape(-1, 3 if np.array(pts).shape[-1] == 3 else 2)
                    if P.shape[1] == 2:
                        P = np.hstack([P, np.zeros((P.shape[0], 1), dtype=np.float32)])
                    if P.shape[0] == 4:
                        obj_points.append(P.reshape(4, 3))
                        ids_list.append(tid)
                    continue
                # Case B: centers only
                if 'tag_id' in item and 'center' in item:
                    if tag_size_m is None:
                        raise RuntimeError("[BOARD] Harvard centers found but tag size unknown. Provide --harvard-tag-size-m or HARVARD_TAG_SIZE_M.")
                    tid = int(item['tag_id'])
                    c = np.array(item['center'], dtype=np.float32).reshape(-1)
                    if center_scale is not None:
                        c = c * float(center_scale)
                    if c.size == 2:
                        cx, cy, cz = float(c[0]), float(c[1]), 0.0
                    else:
                        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
                    half = float(tag_size_m) * 0.5
                    pts3 = np.array([
                        [cx - half, cy - half, cz],
                        [cx + half, cy - half, cz],
                        [cx + half, cy + half, cz],
                        [cx - half, cy + half, cz],
                    ], dtype=np.float32)
                    obj_points.append(pts3.reshape(4, 3))
                    ids_list.append(tid)
                    continue
        else:
            return None

        if not obj_points or not ids_list:
            return None
        ids_arr = np.array(ids_list, dtype=np.int32).reshape(-1, 1)
        try:
            board = cv2.aruco.Board(obj_points, dictionary, ids_arr)
        except Exception:
            try:
                board = cv2.aruco.Board_create(obj_points, dictionary, ids_arr)
            except Exception:
                board = None
        return board
    except Exception:
        try:
            print(f"[BOARD] Failed to parse board; available keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        except Exception:
            pass
        return None


