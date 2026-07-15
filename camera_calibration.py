import numpy as np

# Estimation for a camera with (1280x800) resolution
def get_camera_matrix(frame_width=1280, frame_height=800):

    focal_length = frame_width * (950.0 / 1280.0)   # OV9281 datasheet: 2.85mm / 3um pixels
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

"""
Change to checkerboard camera calibration in the future for more precise
"""