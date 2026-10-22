
import cv2

def get_aruco_detector(dictionary=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)

    params = cv2.aruco.DetectorParameters()

    # Improve detection of smaller markers
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4

    # Allow smaller marker candidates
    params.minMarkerPerimeterRate = 0.015

    # Improve corner precision
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    return detector, aruco_dict
