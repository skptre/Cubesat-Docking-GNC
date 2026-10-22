
import cv2
import numpy as np
import threading
import time
import subprocess

from flask import Flask, Response
from alignment_calculator import AlignmentCalculator
from docking_gui import DockingGUI
from aruco_helper import get_aruco_detector
from face_model import (
    build_board,
    estimate_face_pose,
    target_in_interface_frame,
    face_angles_deg
)

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_DISTANCE = 1.0

CAMERA_DEVICE = "/dev/video0"
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
CAMERA_FPS = 30

EXPOSURE_TIME = 200
CAMERA_GAIN = 20

DEBUG_SAVE_INTERVAL = 3.0

global_frame = None
frame_lock = threading.Lock()


# ==========================================
# CAMERA SETTINGS
# ==========================================
def configure_camera():
    """
    Apply working Arducam B0332 exposure settings
    using V4L2.
    """

    print("[CAMERA] Applying exposure settings...")

    controls = [
        "auto_exposure=1",
        f"exposure_time_absolute={EXPOSURE_TIME}",
        f"gain={CAMERA_GAIN}"
    ]

    for control in controls:
        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                CAMERA_DEVICE,
                f"--set-ctrl={control}"
            ],
            check=True
        )

    print("[CAMERA] Manual exposure enabled")
    print(f"[CAMERA] Exposure: {EXPOSURE_TIME}")
    print(f"[CAMERA] Gain: {CAMERA_GAIN}")


# ==========================================
# CAMERA WORKER
# ==========================================
def camera_worker_thread():
    """
    Read frames from the Arducam USB camera
    using OpenCV and V4L2.
    """

    global global_frame

    print("[CAMERA] Opening Arducam B0332...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():
        print("[CAMERA] FATAL: Could not open camera")
        return

    # Set camera format and resolution
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Apply exposure settings after camera initialization
    try:
        configure_camera()
    except subprocess.CalledProcessError as e:
        print(f"[CAMERA] Warning: Exposure setup failed: {e}")

    print(
        "[CAMERA] Actual resolution:",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "x",
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        "[CAMERA] Actual FPS:",
        cap.get(cv2.CAP_PROP_FPS)
    )

    print("[CAMERA] Streaming frames to GNC loop...")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("[CAMERA] Frame dropped. Retrying...")
                time.sleep(0.05)
                continue

            # Convert grayscale camera frames to BGR
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_GRAY2BGR
                )

            elif (
                len(frame.shape) == 3
                and frame.shape[2] == 1
            ):
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_GRAY2BGR
                )

            with frame_lock:
                global_frame = frame.copy()

    finally:
        cap.release()


# ==========================================
# CAMERA INTRINSICS
# ==========================================
def get_camera_matrix(frame_width, frame_height):

    # Approximate focal length until calibration
    focal_length = frame_width * (950.0 / 1280.0)

    center_x = frame_width / 2
    center_y = frame_height / 2

    return np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)


dist_coeffs = np.zeros((5, 1), dtype=np.float32)


# ==========================================
# ARUCO DEBUG IMAGE SAVING
# ==========================================
def save_debug_images(frame, corners, ids, rejected):

    # Original camera frame
    cv2.imwrite(
        "/tmp/aruco_raw.jpg",
        frame
    )

    debug_frame = frame.copy()

    # Draw detected markers in green
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(
            debug_frame,
            corners,
            ids,
            borderColor=(0, 255, 0)
        )

    # Draw rejected candidates in red
    if len(rejected) > 0:
        cv2.aruco.drawDetectedMarkers(
            debug_frame,
            rejected,
            borderColor=(0, 0, 255)
        )

    detected_ids = (
        ids.flatten().tolist()
        if ids is not None else []
    )

    cv2.putText(
        debug_frame,
        f"Detected: {detected_ids}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        debug_frame,
        f"Rejected: {len(rejected)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imwrite(
        "/tmp/aruco_debug.jpg",
        debug_frame
    )

    print("[DEBUG] Saved ArUco images to /tmp/")


# ==========================================
# TELEMETRY AND DOCKING GUIDANCE
# ==========================================
def generate_telemetry_frames():

    alignment_calc = AlignmentCalculator(
        TARGET_DISTANCE
    )

    detector, aruco_dict = get_aruco_detector(
        cv2.aruco.DICT_4X4_50
    )

    board = build_board(aruco_dict)

    last_debug_save = 0

    while True:

        t0 = time.time()

        # Get latest camera frame
        with frame_lock:
            if global_frame is None:
                frame = None
            else:
                frame = global_frame.copy()

        if frame is None:
            time.sleep(0.1)
            continue

        frame_height, frame_width = frame.shape[:2]

        camera_matrix = get_camera_matrix(
            frame_width,
            frame_height
        )

        gui = DockingGUI(
            frame_width,
            frame_height
        )

        # ==================================
        # ARUCO DETECTION
        # ==================================
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, rejected = (
            detector.detectMarkers(gray)
        )

        detected_ids = (
            ids.flatten().tolist()
            if ids is not None else []
        )

        print(
            f"ids={detected_ids}, "
            f"rejected_count={len(rejected)}"
        )

        t_detect = time.time()

        # ==================================
        # DEBUG IMAGES
        # ==================================
        if (
            time.time() - last_debug_save
            >= DEBUG_SAVE_INTERVAL
        ):

            save_debug_images(
                frame,
                corners,
                ids,
                rejected
            )

            last_debug_save = time.time()

        # ==================================
        # DOCKING POSE ESTIMATION
        # ==================================
        if ids is not None:

            cv2.aruco.drawDetectedMarkers(
                frame,
                corners,
                ids
            )

            rvec, tvec, n_used = estimate_face_pose(
                board,
                corners,
                ids,
                camera_matrix,
                dist_coeffs
            )

            if rvec is not None:

                # Draw docking coordinate axes
                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    0.03
                )

                # Camera-frame position
                pos_cam = tvec.flatten()

                # Interface-frame position
                pos_int = target_in_interface_frame(
                    tvec
                )

                # Face orientation
                yaw, pitch, roll = face_angles_deg(
                    rvec
                )

                # Alignment errors
                error_dict = (
                    alignment_calc.calculate_alignment_error(
                        pos_int
                    )
                )

                # Movement commands
                command_dict = (
                    alignment_calc.get_movement_command(
                        error_dict
                    )
                )

                # Target position on screen
                target_point = gui.draw_target_point(
                    frame,
                    pos_cam,
                    camera_matrix
                )

                # Movement arrows
                gui.draw_alignment_arrows(
                    frame,
                    error_dict,
                    target_point
                )

                # Distance indicator
                gui.draw_distance_indicator(
                    frame,
                    pos_int,
                    TARGET_DISTANCE
                )

                # Status panel
                gui.draw_status_panel(
                    frame,
                    command_dict,
                    error_dict
                )

                # Pose telemetry
                gui.draw_pose_strip(
                    frame,
                    pos_int[2],
                    yaw,
                    pitch,
                    roll,
                    n_used
                )

            else:

                gui.draw_status_panel(
                    frame,
                    {
                        "status": "NO POSE",
                        "command": (
                            "Markers seen, pose solve failed"
                        )
                    },
                    None
                )

        else:

            command_dict = {
                "status": "NO MARKERS",
                "command": "No markers detected"
            }

            gui.draw_status_panel(
                frame,
                command_dict,
                None
            )

        # Draw center crosshair
        gui.draw_crosshair(frame)

        # ==================================
        # FLASK VIDEO STREAM
        # ==================================
        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            print("[ERROR] JPEG encoding failed")
            continue

        frame_bytes = buffer.tobytes()

        t_encode = time.time()

        print(
            f"detect={1000*(t_detect-t0):.1f}ms "
            f"rest={1000*(t_encode-t_detect):.1f}ms"
        )

        dt = time.time() - t0

        if dt > (1.0 / CAMERA_FPS):
            print(
                f"[SLOW FRAME] {dt*1000:.1f}ms"
            )

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ==========================================
# FLASK ROUTE
# ==========================================
@app.route("/")
def video_feed():

    return Response(
        generate_telemetry_frames(),
        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        )
    )


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    print("[SYSTEM] Initializing camera...")

    subprocess.run(
        ["./camera_init.sh"],
        check=True
    )

    time.sleep(1)

    cam_thread = threading.Thread(
        target=camera_worker_thread,
        daemon=True
    )

    cam_thread.start()

    print("=" * 50)
    print("CUBESAT DOCKING GNC TELEMETRY SERVER")
    print("Camera: Arducam B0332")
    print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Requested FPS: {CAMERA_FPS}")
    print(f"Exposure: {EXPOSURE_TIME}")
    print(f"Gain: {CAMERA_GAIN}")
    print("Port: 5000")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        use_reloader=False
    )
