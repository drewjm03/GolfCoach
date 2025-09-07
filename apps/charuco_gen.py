import cv2, numpy as np

# Board parameters (36x24 in poster: ~914x610 mm)
squaresX, squaresY = 12, 9
square_mm = 75.0
marker_mm = 0.7 * square_mm

# Render size at 600 DPI (pixels = mm * dpi / 25.4)
dpi = 600
W_mm = squaresX * square_mm
H_mm = squaresY * square_mm
W_px = int(round(W_mm * dpi / 25.4))
H_px = int(round(H_mm * dpi / 25.4))

aruco = cv2.aruco
dict_id = aruco.DICT_5X5_1000
dictionary = aruco.getPredefinedDictionary(dict_id)
board = aruco.CharucoBoard((squaresX, squaresY),
                           square_mm/1000.0, marker_mm/1000.0,
                           dictionary)
img = board.generateImage((W_px, H_px), marginSize=0, borderBits=1)

# Optional: draw a 100 mm scale bar in a corner
bar_mm = 100.0
bar_px = int(round(bar_mm * dpi / 25.4))
cv2.rectangle(img, (50, H_px-80), (50+bar_px, H_px-60), 0, -1)  # black bar

cv2.imwrite("charuco_12x9_75mm_600dpi.png", img)
print("Saved charuco_12x9_75mm_600dpi.png  (print at 100% scale, matte)")
