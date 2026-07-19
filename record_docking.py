import csv
import os
import sys
import time

import cv2
import numpy as np

from aruco_helper import get_aruco_detector
from face_model import (build_board, estimate_face_pose,
                        target_in_interface_frame, face_angles_deg)

TCP_URL = "tcp://127.0.0.1:8888?listen_timeout=5000"
WIDTH, HEIGHT = 1280, 800
SAVE_EVERY_N = 3          # save every 3rd raw frame; 1 = save all (more disk)
JPEG_QUALITY = 92


def get_camera_matrix(w, h):
    """Same intrinsics path as web_docking: calibration file if present,
    datasheet-derived guess otherwise."""
    try:
        data = np.load("camera_calibration.npz")
        print("[rec] using calibrated intrinsics")
        return data["K"].astype(np.float32), data["dist"].astype(np.float32)
    except (FileNotFoundError, KeyError):
        f = w * (950.0 / 1280.0)   # OV9281 datasheet: 2.85mm / 3um
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
        print("[rec] using datasheet intrinsics (uncalibrated)")
        return K, np.zeros((5, 1), np.float32)


def main():
    if len(sys.argv) > 1:
        session = sys.argv[1]                      # optional custom name
    else:
        n = 1
        while os.path.exists(f"run{n:02d}"):       # find the next free number
            n += 1
        session = f"run{n:02d}"
    if os.path.exists(session):                    
        session += time.strftime("_%H%M%S")
    frames_dir = os.path.join(session, "frames")
    os.makedirs(frames_dir)
    print(f"[rec] recording to {session}/")

    K, dist = get_camera_matrix(WIDTH, HEIGHT)
    detector, aruco_dict = get_aruco_detector(cv2.aruco.DICT_4X4_50)
    board = build_board(aruco_dict)

    cap = cv2.VideoCapture(TCP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[rec] FATAL: camera stream not reachable — is rpicam-vid running?")
        return

    csv_path = os.path.join(session, "pose_log.csv")
    f = open(csv_path, "w", newline="")
    log = csv.writer(f)
    log.writerow(["t_unix", "frame", "n_markers",
                  "x_m", "y_m", "z_m", "yaw_deg", "pitch_deg", "roll_deg",
                  "saved_frame"])

    print(f"[rec] recording to {session}/ — Ctrl-C to stop")
    i = 0
    n_logged = n_saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.time()
            i += 1

            fname = ""
            if i % SAVE_EVERY_N == 0:
                fname = f"{t:.3f}_f{i:06d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, fname), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                n_saved += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
                if frame.ndim == 3 else frame
            corners, ids, _ = detector.detectMarkers(gray)

            row = [f"{t:.4f}", i, 0, "", "", "", "", "", "", fname]
            if ids is not None and len(ids) > 0:
                rvec, tvec, n = estimate_face_pose(board, corners, ids, K, dist)
                if rvec is not None:
                    p = target_in_interface_frame(tvec)
                    yaw, pitch, roll = face_angles_deg(rvec)
                    row = [f"{t:.4f}", i, n,
                           f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}",
                           f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}", fname]
            log.writerow(row)
            n_logged += 1
            if n_logged % 100 == 0:
                f.flush()
                print(f"[rec] {n_logged} frames logged, {n_saved} saved", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        cap.release()
        print(f"\n[rec] done: {n_logged} rows -> {csv_path}, "
              f"{n_saved} frames -> {frames_dir}/")


if __name__ == "__main__":
    main()