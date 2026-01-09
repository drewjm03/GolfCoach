from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def save_npz_compressed(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def load_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}







