import cv2, numpy as np
from . import config
from .calib import StereoSample

try:
    from pupil_apriltags import Detector as PupilDetector
    HAVE_PUPIL = True
except Exception:
    PupilDetector = None
    HAVE_PUPIL = False

def reorder_corners_to_board(corners_1x4x2, obj4x3):
    img = corners_1x4x2.reshape(4,2).astype(np.float32)
    obj = obj4x3[:, :2].astype(np.float32)
    perms = [
        [0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2],
        [0,3,2,1],[3,2,1,0],[2,1,0,3],[1,0,3,2]
    ]
    best_p, best_err = None, 1e9
    for p in perms:
        imgp = img[p]
        H, _ = cv2.findHomography(obj, imgp, 0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(obj.reshape(-1,1,2), H).reshape(-1,2)
        err = float(np.mean(np.linalg.norm(proj - imgp, axis=1)))
        if err < best_err:
            best_err, best_p = err, p
    return img[best_p].reshape(1,4,2) if best_p is not None else corners_1x4x2

def board_ids_safe(board):
    ids = getattr(board, "ids", None)
    if ids is None:
        try:
            ids = board.getIds()
        except Exception:
            ids = None
    if ids is None:
        try:
            N = len(board.getObjPoints())
        except Exception:
            N = 0
        ids = np.arange(N, dtype=np.int32).reshape(-1, 1)
    return np.asarray(ids, dtype=np.int32).reshape(-1, 1)

class CalibrationAccumulator:
    def __init__(self, board, image_size, corner_order_override=None, disable_corner_autoreorder=False):
        self.board = board
        self.image_size = image_size
        self.detector = self._make_detector()
        self._pupil = None
        self.backend_name = "OpenCV ArUco"
        # Optional corner ordering controls
        self.corner_order_override = None
        if isinstance(corner_order_override, (list, tuple)) and len(corner_order_override) == 4:
            try:
                if sorted(list(corner_order_override)) == [0, 1, 2, 3]:
                    self.corner_order_override = [int(x) for x in corner_order_override]
            except Exception:
                self.corner_order_override = None
        self.disable_corner_autoreorder = bool(disable_corner_autoreorder)
        if HAVE_PUPIL:
            try:
                self._pupil = PupilDetector(
                    families=self._apriltag_family_string(),
                    nthreads=2,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=True,
                    decode_sharpening=0.25,
                )
                self.backend_name = "pupil-apriltags"
            except Exception as e:
                print("[APRIL] pupil-apriltags init failed:", e)
        print(f"[APRIL] Detector backend: {self.backend_name}")

        self.corners0, self.ids0, self.counter0 = [], [], []
        self.corners1, self.ids1, self.counter1 = [], [], []
        self.stereo_samples = []

        self.K0 = None; self.D0 = None; self.rms0 = None
        self.K1 = None; self.D1 = None; self.rms1 = None

        self.id_to_obj = self._build_id_to_object()

    def get_backend_name(self):
        return getattr(self, "backend_name", "OpenCV ArUco")

    def _make_detector(self):
        dictionary = cv2.aruco.getPredefinedDictionary(config.APRIL_DICT)
        params = cv2.aruco.DetectorParameters()
        try:
            params.minMarkerPerimeterRate = 0.02
            params.maxMarkerPerimeterRate = 4.0
        except Exception:
            pass
        try:
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 4
        except Exception:
            pass
        try:
            params.adaptiveThreshConstant = 7
        except Exception:
            pass
        try:
            params.perspectiveRemoveIgnoredMarginPerCell = 0.2
            params.perspectiveRemovePixelPerCell = 8
        except Exception:
            pass
        try:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMinAccuracy = 0.01
        except Exception:
            pass
        try:
            params.detectInvertedMarker = True
        except Exception:
            pass
        return cv2.aruco.ArucoDetector(dictionary, params)

    def _apriltag_family_string(self):
        try:
            if config.APRIL_DICT == cv2.aruco.DICT_APRILTAG_36h11:
                return "tag36h11"
            if hasattr(cv2.aruco, 'DICT_APRILTAG_25h9') and config.APRIL_DICT == cv2.aruco.DICT_APRILTAG_25h9:
                return "tag25h9"
            if hasattr(cv2.aruco, 'DICT_APRILTAG_16h5') and config.APRIL_DICT == cv2.aruco.DICT_APRILTAG_16h5:
                return "tag16h5"
            if hasattr(cv2.aruco, 'DICT_APRILTAG_36h10') and config.APRIL_DICT == cv2.aruco.DICT_APRILTAG_36h10:
                return "tag36h10"
        except Exception:
            pass
        return "tag36h11"

    def _build_id_to_object(self):
        id_to_obj = {}
        try:
            ids = board_ids_safe(self.board).flatten().astype(int)
            obj_points = self.board.getObjPoints()
            for idx, tag_id in enumerate(ids):
                obj = np.array(obj_points[idx], dtype=np.float32).reshape(-1, 3)
                id_to_obj[int(tag_id)] = obj
        except Exception as e:
            N = config.TAGS_X * config.TAGS_Y
            try:
                obj_points = self.board.getObjPoints()
                for idx in range(N):
                    obj = np.array(obj_points[idx], dtype=np.float32).reshape(-1, 3)
                    id_to_obj[idx] = obj
            except Exception as inner_e:
                print(f"[ERROR] Could not get object points: {inner_e}")
                id_to_obj = {}
        return id_to_obj

    @staticmethod
    def _avg_side_px(corners_1x4x2):
        c = corners_1x4x2.reshape(4, 2).astype(np.float32)
        return float(sum(np.linalg.norm(c[(i+1) % 4] - c[i]) for i in range(4)) / 4.0)

    def detect(self, gray):
        allowed_ids = set(self.id_to_obj.keys()) if (self.id_to_obj and config.USE_ID_GATING) else None

        if self._pupil is not None:
            for img in (gray, 255 - gray):
                try:
                    dets = self._pupil.detect(img, estimate_tag_pose=False)
                except Exception:
                    dets = []
                corners, ids = [], []
                for d in (dets or []):
                    tid = int(d.tag_id)
                    if getattr(d, "hamming", 0) > config.MAX_HAMMING:
                        continue
                    if getattr(d, "decision_margin", 0.0) < config.MIN_DECISION_MARGIN:
                        continue
                    c = np.array(d.corners, dtype=np.float32).reshape(1, 4, 2)
                    # Apply manual override or auto-reorder to match board convention
                    if self.corner_order_override is not None:
                        c = c[:, self.corner_order_override, :]
                    elif not self.disable_corner_autoreorder and (tid in self.id_to_obj):
                        c = reorder_corners_to_board(c, self.id_to_obj[tid])
                    if self._avg_side_px(c) < config.MIN_SIDE_PX:
                        continue
                    if allowed_ids is not None and tid not in allowed_ids:
                        continue
                    corners.append(c); ids.append([tid])
                if ids:
                    ids = np.array(ids, dtype=np.int32)
                    try:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01)
                        for c in corners:
                            pts = c.reshape(-1,1,2).astype(np.float32)
                            cv2.cornerSubPix(gray, pts, (3,3), (-1,-1), criteria)
                    except Exception:
                        pass
                    return corners, ids

        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return [], None

        filt_c, filt_i = [], []
        for c, i in zip(corners, ids):
            tid = int(i[0])
            if self._avg_side_px(c) < config.MIN_SIDE_PX:
                continue
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            # Normalize corner order to match board's corner convention
            if tid in self.id_to_obj:
                c = reorder_corners_to_board(c, self.id_to_obj[tid])
            filt_c.append(c); filt_i.append([tid])

        if not filt_i:
            return [], None

        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01)
            for c in filt_c:
                pts = c.reshape(-1,1,2).astype(np.float32)
                cv2.cornerSubPix(gray, pts, (3,3), (-1,-1), criteria)
        except Exception:
            pass

        return filt_c, np.array(filt_i, dtype=np.int32)

    def _accumulate_single(self, cam_idx, corners, ids):
        if corners is None or ids is None or len(corners) < config.MIN_MARKERS_PER_VIEW:
            return False
        if cam_idx == 0:
            self.corners0.append(corners)
            self.ids0.append(ids)
            self.counter0.append(len(ids))
        else:
            self.corners1.append(corners)
            self.ids1.append(ids)
            self.counter1.append(len(ids))
        return True

    def _match_stereo(self, corners0, ids0, corners1, ids1):
        map0 = {int(i[0]): c.reshape(-1,2) for c, i in zip(corners0, ids0)}
        map1 = {int(i[0]): c.reshape(-1,2) for c, i in zip(corners1, ids1)}
        common = sorted(set(map0.keys()) & set(map1.keys()))
        if len(common) == 0:
            return None
        obj_pts = []
        img0 = []
        img1 = []
        for tag_id in common:
            if tag_id not in self.id_to_obj:
                continue
            obj = self.id_to_obj[tag_id]
            obj_pts.append(obj)
            img0.append(map0[tag_id])
            img1.append(map1[tag_id])
        if len(obj_pts) == 0:
            return None
        obj_pts = np.concatenate(obj_pts, axis=0).astype(np.float32)
        img0 = np.concatenate(img0, axis=0).astype(np.float32)
        img1 = np.concatenate(img1, axis=0).astype(np.float32)
        return StereoSample(obj_pts, img0, img1)

    def accumulate_pair(self, gray0, gray1):
        c0, i0 = self.detect(gray0)
        c1, i1 = self.detect(gray1)
        ok0 = self._accumulate_single(0, c0, i0)
        ok1 = self._accumulate_single(1, c1, i1)
        if ok0 and ok1:
            sample = self._match_stereo(c0, i0, c1, i1)
            if sample is not None and sample.obj_pts.shape[0] >= config.MIN_MARKERS_PER_VIEW*4:
                self.stereo_samples.append(sample)
                return True
        return False

    def enough_samples(self):
        return (len(self.corners0) >= config.MIN_SAMPLES and
                len(self.corners1) >= config.MIN_SAMPLES and
                len(self.stereo_samples) >= max(8, config.MIN_SAMPLES//2))

    def _mono_calibrate(self, which):
        if which == 0:
            corners, ids, counter = self.corners0, self.ids0, self.counter0
        else:
            corners, ids, counter = self.corners1, self.ids1, self.counter1
        if len(corners) == 0:
            return None, None, None
        # Always use explicit obj/img points to avoid corner-order assumptions
        obj_pts_list = []
        img_pts_list = []
        for corners_img, ids_img in zip(corners, ids):
            obj_pts = []
            img_pts = []
            for c, idv in zip(corners_img, ids_img):
                tag_id = int(idv[0])
                if tag_id not in self.id_to_obj:
                    continue
                obj_pts.append(self.id_to_obj[tag_id])
                img_pts.append(c.reshape(-1,2))
            if len(obj_pts) == 0:
                continue
            obj_pts_list.append(np.concatenate(obj_pts, axis=0).astype(np.float32))
            img_pts_list.append(np.concatenate(img_pts, axis=0).astype(np.float32))
        if len(obj_pts_list) == 0:
            return None, None, None
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((5,1), dtype=np.float64)
        rms, K, D, _, _ = cv2.calibrateCamera(obj_pts_list, img_pts_list, self.image_size, K, D)
        return rms, K, D

    def calibrate_if_possible(self, results):
        changed = False
        if self.K0 is None or self.D0 is None:
            rms0, K0, D0 = self._mono_calibrate(0)
            if K0 is not None:
                self.K0, self.D0, self.rms0 = K0, D0, rms0
                results.K0, results.D0, results.rms0 = K0, D0, rms0
                changed = True
        if self.K1 is None or self.D1 is None:
            rms1, K1, D1 = self._mono_calibrate(1)
            if K1 is not None:
                self.K1, self.D1, self.rms1 = K1, D1, rms1
                results.K1, results.D1, results.rms1 = K1, D1, rms1
                changed = True

        if self.K0 is not None and self.K1 is not None and len(self.stereo_samples) >= 5:
            obj_list = [s.obj_pts for s in self.stereo_samples]
            img0_list = [s.img_pts0 for s in self.stereo_samples]
            img1_list = [s.img_pts1 for s in self.stereo_samples]
            flags = (cv2.CALIB_FIX_INTRINSIC)
            try:
                rms_st, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
                    obj_list, img0_list, img1_list,
                    self.K0.copy(), self.D0.copy(),
                    self.K1.copy(), self.D1.copy(),
                    self.image_size,
                    flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6))
                results.K0, results.D0 = K0, D0
                results.K1, results.D1 = K1, D1
                results.R, results.T, results.E, results.F = R, T, E, F
                results.rms_stereo = float(rms_st)
                changed = True
            except Exception as e:
                print("[CAL] stereoCalibrate failed:", e)
        if results.image_size is None:
            results.image_size = self.image_size
        return changed

# Probe helpers (cached) ---------------------------------
_PROBE_PUPIL = None
def _get_probe_pupil():
    global _PROBE_PUPIL
    if _PROBE_PUPIL is None:
        if not HAVE_PUPIL:
            return None
        _PROBE_PUPIL = PupilDetector(
            families="tag36h11", nthreads=2, quad_decimate=1.0,
            quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25
        )
    return _PROBE_PUPIL

def smoke_test_tag36h11(gray):
    det = _get_probe_pupil()
    if det is None:
        return []
    dets = det.detect(gray, estimate_tag_pose=False) or det.detect(255 - gray, estimate_tag_pose=False)
    ids = []
    for d in (dets or []):
        if getattr(d, "hamming", 0) > 2:
            continue
        if getattr(d, "decision_margin", 0.0) < 15:
            continue
        c = np.array(d.corners, dtype=np.float32).reshape(4, 2)
        side = float(sum(np.linalg.norm(c[(i+1)%4] - c[i]) for i in range(4)) / 4.0)
        if side < 16:
            continue
        ids.append(int(d.tag_id))
    return sorted(set(ids))

_ARUCO_PROBE_CACHE = {}
def _get_probe_aruco(det_code):
    if det_code in _ARUCO_PROBE_CACHE:
        return _ARUCO_PROBE_CACHE[det_code]
    D = cv2.aruco
    dic = D.getPredefinedDictionary(det_code)
    params = D.DetectorParameters()
    try:
        params.detectInvertedMarker = True
    except Exception:
        pass
    det = D.ArucoDetector(dic, params)
    _ARUCO_PROBE_CACHE[det_code] = det
    return det

def probe_aruco_6x6(gray):
    D = cv2.aruco
    results = {}
    for name, code in [("DICT_6X6_50", D.DICT_6X6_50),
                       ("DICT_6X6_100", D.DICT_6X6_100),
                       ("DICT_6X6_250", D.DICT_6X6_250),
                       ("DICT_6X6_1000", D.DICT_6X6_1000)]:
        det = _get_probe_aruco(code)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            results[name] = int(len(ids))
    return results


