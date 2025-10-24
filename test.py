import cv2
from cv2 import aruco

dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

board = aruco.GridBoard(
    markersX=5,
    markersY=7,
    markerLength=0.04,
    markerSeparation=0.01,
    dictionary=dict_aruco
)

print("✅ GridBoard created successfully!")