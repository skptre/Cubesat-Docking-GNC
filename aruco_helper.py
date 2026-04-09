import cv2
import numpy as np

def estimate_pose_single_markers(corners, marker_size, camera_matrix, dist_coeffs):

    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0],
    ], dtype=np.float32)

    rvecs = []
    tvecs = []

    for corner in corners:
        # SolvePnP for pose estimation
        _, rvec, tvec = cv2.solvePnP(
            marker_points,
            corner,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(rvec)
        tvecs.append(tvec)

    # Convert to numpy arrays
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    return rvecs, tvecs, None

def get_aruco_detector(dictionary=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    aruco_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return detector, aruco_dict
