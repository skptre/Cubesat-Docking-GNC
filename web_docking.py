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
TARGET_DISTANCE = 1.0 # 1m station-keeping
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
    cv2.putText(frame, "CRITICAL ERROR:", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, error_message, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_telemetry_frames():
    #Failsafe 1: Camera hardware not found or picamera2 not installed
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize. Error: {e}")
        yield get_error_frame("CAM INIT FAILED (CHECK PICAMERA2)")
        return

    #Failsafe 2: Camera found but cannot configure the resolution or start the stream
    try:
        # Request a standard size and let the Pi ISP convert the mono to RGB
        # This prevents OpenCV from crashing on 10-bit raw data
        config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 400)})
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(f"CRITICAL: Failed to start stream. Error: {e}")
        yield get_error_frame("STREAM START FAILED")
        return

    # Initialize GNC Math and GUI
    camera_matrix = get_camera_matrix(640, 480)
    alignment_calc = AlignmentCalculator(TARGET_DISTANCE, REQUIRED_MARKERS)
    gui = DockingGUI(640, 480)
    detector, aruco_dict = get_aruco_detector(cv2.aruco.DICT_4X4_50)

    while True:
        #Failsafe 3: Camera is running but drops a frame or gets unplugged
        try:
            frame = picam2.capture_array()
            # Picamera2 outputs RGB, OpenCV expects BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"WARNING: Camera feed lost or failed to grab frame. Error: {e}")
            yield get_error_frame("FRAME CAPTURE FAILED")
            break

        #GNC Logic
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

        # Encode the drawn frame to a memory buffer and push to web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_telemetry_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("=" * 50)
    print("TELEMETRY SERVER ONLINE (PICAMERA2 ENGINE)")
    print("Port: 5000")
    print("=" * 50)
    # Run server without the Pi 5 crashing
    app.run(host='0.0.0.0', port=5000, threaded=True)