import cv2
import numpy as np
from aruco_helper import *

# 4x4 markers with 50 different IDs
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Detector object
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def get_camera_matrix(frame_width, frame_height):

    focal_length = frame_width * 0.7
    center_x = frame_width / 2
    center_y = frame_height / 2

    camera_matrix = np.array([ 
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    return camera_matrix

# Distortion coefficients
dist_coeffs = np.zeros((5, 1), np.float32)

# Marker size in meters
MARKER_SIZE = 0.01 # Change to actual size of markers

capture = cv2.VideoCapture(0)

# Get frame dimensions
ret, frame = capture.read()
frame_height, frame_width = frame.shape[:2]
camera_matrix = get_camera_matrix(frame_width, frame_height)


if not capture.isOpened():
    print("Could not open webcam.")
    exit()

print("Starting pose estimation. Press 'q' to quit.")
print(f"Camera resolution: {frame_width}x{frame_height}")
print(f"Marker size: {MARKER_SIZE}m")

while True:
    ret, frame = capture.read()
    if not ret:
        print("Frame not captured. Exiting.")

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Draw markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose for each marker
        rvecs, tvecs, _ = estimate_pose_single_markers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

        # X =  red, Y = green, Z = blue
        for i in range(len(ids)):

            # Offset to bottom-left corner
            tvec_corner = tvecs[i].copy()
            R, _ = cv2.Rodrigues(rvecs[i])
            corner_offset = np.array([[-MARKER_SIZE / 2], [-MARKER_SIZE / 2], [0]])
            tvec_corner += R @ corner_offset


            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvecs[i], tvec_corner, MARKER_SIZE)

        print(f"Detected {len(ids)} markers: {ids.flatten()}")

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


