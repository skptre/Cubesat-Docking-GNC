import cv2
import numpy as np
from alignment_calculator import AlignmentCalculator
from docking_gui import DockingGUI
from aruco_helper import *

# ArUco setup
detector, aruco_dict = get_aruco_detector(cv2.aruco.DICT_4X4_50)

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

dist_coeffs = np.zeros((5,1), dtype=np.float32)

# Configuration
MARKER_SIZE = 0.05 # 5cm markers
TARGET_DISTANCE = 0.3 # 30cm from marker
REQUIRED_MARKERS = [0, 1, 2, 3]

# Initialize
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
camera_matrix = get_camera_matrix(frame_width, frame_height)

alignment_calc = AlignmentCalculator(TARGET_DISTANCE, REQUIRED_MARKERS)
gui = DockingGUI(frame_width, frame_height)

print("=" * 60)
print("SATELLITE DOCKING ALIGNMENT SYSTEM")
print("=" * 60)
print(f"Camera Resolution: {frame_width}x{frame_height}")
print(f"Marker Size: {MARKER_SIZE * 100}cm")
print(f"Target Distance: {TARGET_DISTANCE * 100}cm")
print(f"Required Markers: {REQUIRED_MARKERS}")
print("\nControls:")
print("  'q' - Quit")
print("  's' - Save screenshot")
print("=" * 60)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = estimate_pose_single_markers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Offset to bottom-left corner
            tvec_corner = tvecs[i].copy()
            R, _ = cv2.Rodrigues(rvecs[i])
            corner_offset = np.array([[-MARKER_SIZE / 2], [-MARKER_SIZE / 2], [0]])
            tvec_corner += R @ corner_offset

            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvecs[i], tvec_corner, MARKER_SIZE)

        # Calculate alignment
        center_position = alignment_calc.calculate_target_center(ids, tvecs)
        error_dict = alignment_calc.calculate_alignment_error(center_position)
        command_dict = alignment_calc.get_movement_command(error_dict)

        # Draw GUI elements
        if center_position is not None:  # Add this check!
            target_point = gui.draw_target_point(frame, center_position, camera_matrix)
            gui.draw_alignment_arrows(frame, error_dict, target_point)
            gui.draw_distance_indicator(frame, center_position, TARGET_DISTANCE)
            gui.draw_status_panel(frame, command_dict, error_dict)
        else:
            # Show which markers are missing
            detected_ids = set(ids.flatten())
            missing_ids = set(REQUIRED_MARKERS) - detected_ids
            command_dict = {
                'status': 'INCOMPLETE',
                'command': f'Missing markers: {sorted(missing_ids)}'
            }
            gui.draw_status_panel(frame, command_dict, None)

    else:
        command_dict = {'status': 'NO MARKERS', 'command': 'No markers detected'}
        gui.draw_status_panel(frame, command_dict, None)


    gui.draw_crosshair(frame)

    cv2.imshow('Satellite Docking System', frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f'docking_screenshot_{frame_count}.png'
        cv2.imwrite(filename, frame)
        print(f'Screenshot saved: {filename}')

cap.release()
cv2.destroyAllWindows()
print("\nSystem shut down.")