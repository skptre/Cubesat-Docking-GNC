import cv2
import numpy as np
from flask import Flask, Response
from alignment_calculator import AlignmentCalculator
from docking_gui import DockingGUI
from aruco_helper import *

# Initialize Flask App
app = Flask(__name__)

# System Configuration
MARKER_SIZE = 0.01
TARGET_DISTANCE = 0.3
REQUIRED_MARKERS = [0, 1, 2, 3]

def get_camera_matrix(frame_width, frame_height):
    focal_length = frame_width * 0.7
    center_x = frame_width / 2
    center_y = frame_height / 2
    return np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

dist_coeffs = np.zeros((5,1), dtype=np.float32)

def get_error_frame(error_message):
    """Generates a black frame with red error text to push to the web dashboard."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw error text
    cv2.putText(frame, "CRITICAL ERROR:", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, error_message, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Encode and format for the Flask web stream
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_telemetry_frames():
    # Force V4L2 backend and lower resolution for Pi CPU efficiency
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # FAILSAFE 1: Camera hardware not found at all
    if not cap.isOpened():
        print("CRITICAL: Camera hardware not found or inaccessible.")
        yield get_error_frame("CAMERA NOT DETECTED (CHECK WIRING)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # FAILSAFE 2: Read first frame to ensure sensor is transmitting
    ret, frame = cap.read()
    if not ret:
        print("CRITICAL: Camera opened but failed to read frame.")
        yield get_error_frame("FAILED TO READ SENSOR DATA")
        return

    frame_height, frame_width = frame.shape[:2]
    camera_matrix = get_camera_matrix(frame_width, frame_height)
    
    alignment_calc = AlignmentCalculator(TARGET_DISTANCE, REQUIRED_MARKERS)
    gui = DockingGUI(frame_width, frame_height)
    detector, aruco_dict = get_aruco_detector(cv2.aruco.DICT_4X4_50)

    while True:
        ret, frame = cap.read()
        
        # FAILSAFE 3: Camera unplugged or failed mid-operation
        if not ret:
            print("WARNING: Camera feed lost during operation.")
            yield get_error_frame("CAMERA CONNECTION LOST")
            break

        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = estimate_pose_single_markers(
                corners, MARKER_SIZE, camera_matrix, dist_coeffs
            )

            for i in range(len(ids)):
                tvec_corner = tvecs[i].copy()
                R, _ = cv2.Rodrigues(rvecs[i])
                corner_offset = np.array([[-MARKER_SIZE / 2], [-MARKER_SIZE / 2], [0]])
                tvec_corner += R @ corner_offset
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i], tvec_corner, MARKER_SIZE)

            center_position = alignment_calc.calculate_target_center(ids, tvecs)
            error_dict = alignment_calc.calculate_alignment_error(center_position)
            command_dict = alignment_calc.get_movement_command(error_dict)

            if center_position is not None:
                target_point = gui.draw_target_point(frame, center_position, camera_matrix)
                gui.draw_alignment_arrows(frame, error_dict, target_point)
                gui.draw_distance_indicator(frame, center_position, TARGET_DISTANCE)
                gui.draw_status_panel(frame, command_dict, error_dict)
            else:
                detected_ids = set(ids.flatten())
                missing_ids = set(REQUIRED_MARKERS) - detected_ids
                command_dict = {
                    'status': 'INCOMPLETE',
                    'command': f'Missing: {sorted(missing_ids)}'
                }
                gui.draw_status_panel(frame, command_dict, None)
        else:
            command_dict = {'status': 'NO MARKERS', 'command': 'No markers detected'}
            gui.draw_status_panel(frame, command_dict, None)

        gui.draw_crosshair(frame)

        # Instead of cv2.imshow, we encode the drawn frame to a memory buffer
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_telemetry_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 50)
    print("TELEMETRY SERVER ONLINE")
    print("Port: 5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, threaded=True)