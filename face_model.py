"""
Return the pose of the face - docking face as ONE rigid object
"""

import cv2
import numpy as np

MARKER_SIZE = 0.02
MARKER_CENTERS = {
    0: (-0.037, +0.037), # Measure on the interface
    1: (+0.037, +0.037), # Measure on the interface
    2: (-0.037, -0.037), # Measure on the interface
    3: (+0.037, -0.037), # Measure on the interface
}

# CAMERA OFFSET
CAMERA_IN_INTERFACE_FRAME = np.array([-0.0475, 0.0, 0.0])    # 47.5mm

# For each marker we generate its 4 corner coordinates in the face frame, in the corner order ArUco detects them: top-left, top-right, bottom-right, bottom-left
def build_board(aruco_dict):
    half = MARKER_SIZE / 2.0
    obj_points = []
    ids = []
    for marker_id, (cx, cy) in MARKER_CENTERS.items():
        corners = np.array([
             [cx - half, cy + half, 0.0],   # top-left
            [cx + half, cy + half, 0.0],   # top-right
            [cx + half, cy - half, 0.0],   # bottom-right
            [cx - half, cy - half, 0.0],   # bottom-left
        ], dtype=np.float32)
        obj_points.append(corners)
        ids.append(marker_id)
        
    return cv2.aruco.Board(
        np.array(obj_points, dtype=np.float32),
        aruco_dict,
        np.array(ids, dtype=np.int32),
     )

# Pose of the docking face from whatever markers are available
def estimate_face_pose(board, corners, ids, camera_matrix, dist_coeffs):

    if ids is None or len(ids) == 0:
        return None, None, 0
    
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:  # need >= 4 point pairs
        return None, None, 0
    
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, 0
    
    return rvec, tvec, len(ids)

# Expressing the error relative to the docking interface, rather than the lens
def target_in_interface_frame(tvec):
    return tvec.flatten() + CAMERA_IN_INTERFACE_FRAME


# Attitude (yaw, pitch, roll)
def face_angles_deg(rvec):

    R, _ = cv2.Rodrigues(rvec)
    # A face pointing straight at the camera has its +z toward the camera, i.e. along the camera's -z. Undo that nominal flip so "aligned" = 0.
    R_align = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    R_rel = R_align.T @ R

    yaw   = np.degrees(np.arctan2(-R_rel[2, 0], R_rel[2, 2]))
    pitch = np.degrees(np.arctan2(R_rel[2, 1], R_rel[2, 2]))
    roll  = np.degrees(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
    return yaw, pitch, roll


