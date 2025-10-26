import cv2, numpy as np

def write_marker(dict_id, marker_id=0, side=600, code_size=6, out="marker.png"):
    """
    Generate a marker using OpenCV's current API.
    Adds a white quiet zone ~1 module wide around the marker.

    dict_id: e.g., cv2.aruco.DICT_APRILTAG_36h11 or cv2.aruco.DICT_6X6_100
    marker_id: integer tag id
    side: inner marker size in pixels (before the border)
    code_size: bit-grid size (36h11 & 6x6 -> 6)
    out: output filename
    """
    D = cv2.aruco.getPredefinedDictionary(dict_id)

    # Use new API if available; otherwise fall back to drawMarker
    try:
        img = cv2.aruco.generateImageMarker(D, marker_id, side)
    except AttributeError:
        img = np.zeros((side, side), dtype=np.uint8)
        cv2.aruco.drawMarker(D, marker_id, side, img, 1)

    # Add quiet zone (one module wide) so detectors accept it
    q = max(8, side // code_size)  # ~1 module
    img = cv2.copyMakeBorder(img, q, q, q, q, cv2.BORDER_CONSTANT, value=255)

    cv2.imwrite(out, img)
    print("Wrote", out)

# Example: AprilTag 36h11, id=0
write_marker(cv2.aruco.DICT_APRILTAG_36h11, marker_id=0, side=600, code_size=6, out="tag36h11_id0.png")

# (Optional) Example: ArUco 6x6, id=23
# write_marker(cv2.aruco.DICT_6X6_100, marker_id=23, side=600, code_size=6, out="aruco6x6_id23.png")
