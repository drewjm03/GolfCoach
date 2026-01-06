import pickle
import joblib

def load_pkl(path):
    # PHALP/4DHumans outputs are commonly joblib-compressed pickles.
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def summarize(obj, name="pkl"):
    print(f"\n=== {name} type={type(obj)} ===")
    if isinstance(obj, dict):
        print("top keys:", list(obj.keys())[:50])
        for k in ["verts", "vertices", "smpl", "smplx", "betas", "pose", "body_pose", "global_orient", "transl", "translation"]:
            if k in obj:
                v = obj[k]
                print(f"  has {k}: type={type(v)}")
    elif isinstance(obj, (list, tuple)):
        print("len:", len(obj))
        if len(obj):
            print("elem0 type:", type(obj[0]))
    else:
        print(obj)

A = load_pkl("runs/my_swing_stageA_120front/less_clutter120cam1.pkl")
B = load_pkl("runs/my_swing_stageA_120front/less_clutter120cam2.pkl")
summarize(A, "A")
summarize(B, "B")

import numpy as np

def peek_one(pkl_dict, name="A"):
    k = sorted(pkl_dict.keys())[0]
    v = pkl_dict[k]
    print(f"\n{name}: first key = {k}")
    print(f"{name}: value type =", type(v))

    if isinstance(v, dict):
        print(f"{name}: value keys =", list(v.keys())[:80])
        # print shapes/types for common fields if present
        for kk in ["verts","vertices","smpl","smplx","smpl_params","pose","body_pose","betas","global_orient","transl","track_id","id","scores","bbox","keypoints","joints3d"]:
            if kk in v:
                x = v[kk]
                try:
                    arr = np.asarray(x)
                    print(f"  {kk}: {type(x)} shape={arr.shape} dtype={arr.dtype}")
                except Exception:
                    print(f"  {kk}: {type(x)}")
    elif isinstance(v, (list, tuple)):
        print(f"{name}: list len =", len(v))
        if len(v):
            e0 = v[0]
            print(f"{name}: elem0 type =", type(e0))
            if isinstance(e0, dict):
                print(f"{name}: elem0 keys =", list(e0.keys())[:80])
    else:
        print(v)

peek_one(A, "A")
peek_one(B, "B")


import numpy as np

def summarize_smpl0(pkl_dict, name="A"):
    k = sorted(pkl_dict.keys())[0]
    frame = pkl_dict[k]
    s0 = frame["smpl"][0]
    print(f"\n{name} smpl[0] type:", type(s0))
    if isinstance(s0, dict):
        print(f"{name} smpl[0] keys:", list(s0.keys())[:80])
        for kk in ["vertices","verts","v","faces","f","betas","body_pose","global_orient","transl",
                   "pose","full_pose","joints","joints3d","cam","camera"]:
            if kk in s0:
                x = s0[kk]
                try:
                    arr = np.asarray(x)
                    print(f"  {kk}: shape={arr.shape} dtype={arr.dtype}")
                except Exception:
                    print(f"  {kk}: type={type(x)}")
    else:
        print("smpl[0] is not a dict; repr:", repr(s0)[:200])

summarize_smpl0(A, "A")
summarize_smpl0(B, "B")
