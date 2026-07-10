import cv2
import numpy as np

def get_aruco_detector(dictionary=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    aruco_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return detector, aruco_dict
