import numpy as np

class CalibrationResults:
    def __init__(self):
        self.K0 = None
        self.D0 = None
        self.K1 = None
        self.D1 = None
        self.image_size = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.rms0 = None
        self.rms1 = None
        self.rms_stereo = None

    def is_complete(self):
        return all([x is not None for x in (self.K0, self.D0, self.K1, self.D1, self.R, self.T)])

    def to_json_dict(self):
        def as_list(x):
            return (x.tolist() if isinstance(x, np.ndarray) else x)
        return {
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "K0": as_list(self.K0),
            "D0": as_list(self.D0),
            "K1": as_list(self.K1),
            "D1": as_list(self.D1),
            "R": as_list(self.R),
            "T": as_list(self.T),
            "E": as_list(self.E),
            "F": as_list(self.F),
            "rms0": self.rms0,
            "rms1": self.rms1,
            "rms_stereo": self.rms_stereo,
        }

class StereoSample:
    def __init__(self, obj_pts, img_pts0, img_pts1):
        self.obj_pts = obj_pts  # (N,3)
        self.img_pts0 = img_pts0  # (N,2)
        self.img_pts1 = img_pts1  # (N,2)


