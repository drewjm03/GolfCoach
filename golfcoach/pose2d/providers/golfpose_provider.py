from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from mmengine.registry import init_default_scope

# mmpose 1.x API
from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples

# mmdet 3.x API (optional)
try:
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules as _register_mmdet_modules

    # Ensure MMDet transforms (e.g., PackDetInputs) and models are registered
    _register_mmdet_modules()
    _HAS_MMDET = True
except Exception:
    _HAS_MMDET = False


@dataclass
class DetectorConfig:
    config: str
    checkpoint: str
    device: str = "cpu"
    score_thr: float = 0.3
    class_id: int = 0            # for 1-class detector
    ema: float = 0.8             # bbox smoothing
    pad_scale: float = 1.25      # expand bbox


class GolfPose2DProvider:
    """
    Runs GolfPose-2D (GC) on one frame given a bbox ROI.
    Optionally runs a detector to get that bbox.
    """

    def __init__(
        self,
        pose_config: str,
        pose_checkpoint: str,
        device: str = "cpu",
        detector: Optional[DetectorConfig] = None,
    ) -> None:
        self.pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
        self.pose_model.cfg.visualizer = None  # avoid GUI

        self.detector_cfg = detector
        self.det_model = None
        if detector is not None:
            if not _HAS_MMDET:
                raise RuntimeError("mmdet is not importable but detector config was provided.")
            self.det_model = init_detector(detector.config, detector.checkpoint, device=detector.device)

        # Read keypoint names from dataset meta if available (recommended to verify ordering!)
        meta = getattr(self.pose_model, "dataset_meta", None) or {}
        id2name = meta.get("keypoint_id2name", None)
        if id2name is None:
            # fallback placeholder; you SHOULD replace these after inspecting meta
            self.joint_names = [f"kpt_{i}" for i in range(22)]
        else:
            self.joint_names = [id2name[i] for i in range(len(id2name))]

        self._prev_bbox: Optional[np.ndarray] = None

    def _expand_bbox(self, bbox: np.ndarray, W: int, H: int, pad_scale: float) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(np.float64)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1) * pad_scale
        h = (y2 - y1) * pad_scale
        x1n = np.clip(cx - 0.5 * w, 0, W - 1)
        y1n = np.clip(cy - 0.5 * h, 0, H - 1)
        x2n = np.clip(cx + 0.5 * w, 0, W - 1)
        y2n = np.clip(cy + 0.5 * h, 0, H - 1)
        return np.array([x1n, y1n, x2n, y2n], dtype=np.float64)

    def _detect_bbox(self, frame_bgr: np.ndarray) -> np.ndarray:
        H, W = frame_bgr.shape[:2]
        assert self.det_model is not None and self.detector_cfg is not None

        # Ensure MMDet's own registry/scope is active for the detector pipeline
        init_default_scope("mmdet")
        det = inference_detector(self.det_model, frame_bgr)
        inst = det.pred_instances

        # Debug info about detector outputs
        boxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy()
        print(
            "DET:",
            boxes.shape,
            "score max:",
            float(scores.max()) if scores.size else None,
            "unique labels:",
            np.unique(labels) if labels.size else None,
        )

        # If detector produced no boxes at all, try to keep the previous ROI.
        if boxes.shape[0] == 0:
            if self._prev_bbox is not None:
                return self._prev_bbox.copy()
            # First-frame (or no history) fallback: full-frame, padded once.
            bbox = np.array([0, 0, W - 1, H - 1], dtype=np.float64)
            bbox = self._expand_bbox(bbox, W, H, self.detector_cfg.pad_scale)
            self._prev_bbox = bbox
            return bbox

        # pick best bbox for the target class
        mask = (labels == int(self.detector_cfg.class_id)) & (scores >= float(self.detector_cfg.score_thr))
        if not np.any(mask):
            # fallback: full frame
            bbox = np.array([0, 0, W - 1, H - 1], dtype=np.float64)
        else:
            idx = np.argmax(scores[mask])
            bbox = boxes[mask][idx].astype(np.float64)

        # Sanity filter: ignore implausibly tiny boxes that can hijack tracking.
        x1, y1, x2, y2 = bbox
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        frame_area = float(W * H)
        if area < 0.02 * frame_area:  # < 2% of image area
            if self._prev_bbox is not None:
                return self._prev_bbox.copy()

        bbox = self._expand_bbox(bbox, W, H, self.detector_cfg.pad_scale)

        # EMA smoothing
        if self._prev_bbox is None:
            self._prev_bbox = bbox
        else:
            a = float(self.detector_cfg.ema)
            self._prev_bbox = a * self._prev_bbox + (1.0 - a) * bbox

        return self._prev_bbox.copy()

    def infer_frame(self, frame_bgr: np.ndarray, bbox: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (kpts_22x2, conf_22, bbox_used_4).
        If bbox is None and detector exists, will run detector.
        If bbox is None and no detector, uses full frame bbox.
        """
        H, W = frame_bgr.shape[:2]

        if bbox is None:
            if self.det_model is not None:
                bbox = self._detect_bbox(frame_bgr)
            else:
                bbox = np.array([0, 0, W - 1, H - 1], dtype=np.float64)

        # mmpose 1.3.x: prefer bboxes array API over raw person dicts
        bboxes = np.array([bbox], dtype=np.float32)  # (1,4)

        # Switch back to MMPose scope for pose model inference
        init_default_scope("mmpose")
        results = inference_topdown(self.pose_model, frame_bgr, bboxes)

        merged = merge_data_samples(results)
        pred = merged.pred_instances

        # First index: (J,2) and (J,) for single person
        kpts = pred.keypoints[0]
        conf = pred.keypoint_scores[0]

        # Robust conversion to numpy (supports tensor or ndarray backends)
        if hasattr(kpts, "detach"):
            kpts = kpts.detach().cpu().numpy()
        else:
            kpts = np.asarray(kpts)

        if hasattr(conf, "detach"):
            conf = conf.detach().cpu().numpy()
        else:
            conf = np.asarray(conf)

        return kpts.astype(np.float32), conf.astype(np.float32), bbox.astype(np.float32)


