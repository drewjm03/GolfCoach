import cv2

# ---------- Tunable constants ----------
CAPTURE_FPS = 60
MAX_COMBINED_WIDTH = 1920
FIRST_FRAME_RETRY_COUNT = 5
WB_TOGGLE_DELAY_S = 0.075
PRESERVE_NATIVE_RES = True  # if True, do not downscale combined preview; keep exact pixel size
DEBUG_PROBES = False        # gate expensive probe/smoke helpers

# Exposure/gain defaults
USE_SDK_EXPOSURE = False
DEFAULT_EXPOSURE_STEP = -7
MIN_EXPOSURE_STEP = -14
MAX_EXPOSURE_STEP = 0
MIN_EXPOSURE_US = 50
SAFETY_EXPOSURE_HEADROOM_US = 100
AUTO_EXPOSURE_COMP_DELTA_US = 50
DEFAULT_WB_KELVIN = 3950
MIN_GAIN = 0.0
MAX_GAIN = 255.0
GAIN_DELTA = 1.0

# AprilTag grid board configuration
APRIL_DICT = cv2.aruco.DICT_APRILTAG_36h11
TAGS_X = 7                # number of tags horizontally (columns)
TAGS_Y = 5                # number of tags vertically (rows)
TAG_SIZE_M = 0.075        # tag black square size in meters
TAG_SEP_M = 0.01875       # white gap between tags in meters

# AprilTag quality / gating
MAX_HAMMING = 0            # only perfect decodes
MIN_DECISION_MARGIN = 30   # raise if you still see clutter
MIN_SIDE_PX = 32           # ignore tiny tags; adjust for your resolution
USE_ID_GATING = False

# Calibration parameters/criteria
MIN_MARKERS_PER_VIEW = 8
MIN_SAMPLES = 20
TARGET_RMS_PX = 0.6
RECALC_INTERVAL_S = 1.0
CALIB_SAMPLE_PERIOD_S = 3.0  # sample cadence in seconds

SENSOR_ROTATE_180 = False


