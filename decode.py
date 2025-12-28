import numpy as np
from pycocotools import mask as mask_utils

rle = f["mask"][0]
mask = mask_utils.decode(rle)
print("decoded mask shape:", mask.shape, "dtype:", mask.dtype, "min/max:", mask.min(), mask.max())

if mask.ndim == 3:
    mask = mask[:, :, 0]
print("final mask shape:", mask.shape, "unique:", np.unique(mask)[:10])
