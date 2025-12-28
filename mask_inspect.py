import numpy as np
from golfcoach.io.phalp_pkl import load_phalp_tracks

pkl_path = r"runs\my_swing_stageA_120front\less_clutter120cam1.pkl"
tracks = load_phalp_tracks(pkl_path)

frame_idx = sorted(tracks.keys())[0]
f = tracks[frame_idx]

print("frame_idx:", frame_idx)
print("frame keys:", list(f.keys()))

# pose vector
pose = f["pose"][0]
print("\npose type:", type(pose), "shape:", getattr(pose, "shape", None), "dtype:", getattr(pose, "dtype", None))
print("pose first 12:", np.asarray(pose).ravel()[:12])

# smpl dict
smpl = f["smpl"][0]
print("\nsmpl type:", type(smpl))
print("smpl keys:", list(smpl.keys()))
for k in smpl.keys():
    v = smpl[k]
    a = np.asarray(v) if not isinstance(v, dict) else v
    if isinstance(a, dict):
        print(" ", k, "-> dict keys:", list(a.keys())[:20])
    else:
        print(" ", k, "->", type(v), "shape:", getattr(a, "shape", None), "dtype:", getattr(a, "dtype", None))

# camera params PHALP predicts
cam = f["camera"][0]
cam_bbox = f["camera_bbox"][0]
print("\ncamera:", np.asarray(cam).shape, np.asarray(cam))
print("camera_bbox:", np.asarray(cam_bbox).shape, np.asarray(cam_bbox))
print("center:", np.asarray(f["center"][0]))
print("scale:", f["scale"][0], "size:", f["size"][0])

import numpy as np
from pycocotools import mask as mask_utils

rle = f["mask"][0]
mask = mask_utils.decode(rle)
print("decoded mask shape:", mask.shape, "dtype:", mask.dtype, "min/max:", mask.min(), mask.max())

if mask.ndim == 3:
    mask = mask[:, :, 0]
print("final mask shape:", mask.shape, "unique:", np.unique(mask)[:10])

j3d = np.asarray(f["3d_joints"][0], dtype=np.float32)  # (45,3)
print("j3d shape:", j3d.shape)
print("j3d min:", np.nanmin(j3d, axis=0))
print("j3d max:", np.nanmax(j3d, axis=0))
print("j3d mean:", np.nanmean(j3d, axis=0))
print("median bone-ish length (rough):", np.nanmedian(np.linalg.norm(np.diff(j3d[:10], axis=0), axis=1)))