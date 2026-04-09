import cv2
import numpy as np
from alignment_calculator import AlignmentCalculator

class DockingGUI:

    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2

        # Colors (BGR format)
        self.COLOR_ALIGNED = (0, 255, 0)  # Green
        self.COLOR_ALIGNING = (0, 165, 255)  # Orange
        self.COLOR_NO_MARKERS = (0, 0, 255)  # Red
        self.COLOR_TARGET = (255, 0, 255)  # Magenta

    def draw_crosshair(self, frame):
        # Draw center crosshair (current camera center)
        cv2.line(frame, (self.center_x - 10, self.center_y),
                 (self.center_x + 10, self.center_y), (255, 255, 255), 1)
        cv2.line(frame, (self.center_x, self.center_y - 10),
                 (self.center_x, self.center_y + 10), (255, 255, 255), 1)
        # cv2.circle(frame, (self.center_x, self.center_y), 5, (255, 255, 255), -1)

    def draw_target_point(self, frame, center_position, camera_matrix):

        if center_position is None:
            return None

        # Project 3D Center point to 2d Image coordinates
        center_3d = center_position.reshape(1,1,3)
        center_2d, _ = cv2.projectPoints(
            center_3d,
            np.zeros((3,1)), np.zeros((3,1)), # No rotation/translation
            camera_matrix,
            np.zeros((5,1))
        )

        target_x = int(center_2d[0][0][0])
        target_y = int(center_2d[0][0][1])

        #Draw target circle
        cv2.circle(frame, (target_x, target_y), 15, self.COLOR_TARGET, 3)
        cv2.circle(frame, (target_x, target_y), 3, self.COLOR_TARGET, -1)

        return target_x, target_y


    def draw_alignment_arrows(self, frame, error_dict, target_point):
        if error_dict is None or target_point is None:
            return

        target_x, target_y = target_point
        arrow_color = self.COLOR_ALIGNING

        # Horizontal arrow
        if abs(error_dict['error_x']) > 0.01:
            start_x = self.center_x
            end_x = target_x
            y = self.center_y
            cv2.arrowedLine(frame, (start_x, y), (end_x, y),
                            arrow_color, 3, tipLength=0.3)

        # Vertical arrow
        if abs(error_dict['error_y']) > 0.01:
            x = self.center_x
            start_y = self.center_y
            end_y = target_y
            cv2.arrowedLine(frame, (x, start_y), (x, end_y),
                            arrow_color, 3, tipLength=0.3)

    def draw_status_panel(self, frame, command_dict, error_dict):

        if command_dict is None:
            return

        # Background panel
        panel_height = 150
        overlay = frame.copy()

        # Status color
        if command_dict['status'] == 'ALIGNED':
            color = self.COLOR_ALIGNED
        elif command_dict['status'] == 'ALIGNING':
            color = self.COLOR_ALIGNING
        else:
            color = self.COLOR_NO_MARKERS

        # Status text
        cv2.putText(frame, f"Status: {command_dict['status']}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Command: {command_dict['command']}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Error details
        if error_dict is not None:
            error_text = (f"Error - X: {error_dict['error_x'] * 100:.1f}cm, "
                          f"Y: {error_dict['error_y'] * 100:.1f}cm, "
                          f"Z: {error_dict['error_z'] * 100:.1f}cm")
            cv2.putText(frame, error_text,
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            total_error_text = f"Total Error: {error_dict['total_error'] * 100:.1f}cm"
            cv2.putText(frame, total_error_text,
                        (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def draw_distance_indicator(self, frame, center_position, target_distance):
        if center_position is None:
            return

        current_z = center_position[2]

        # Distance bar
        bar_x = self.width - 50
        bar_y_start = 200
        bar_y_end = self.height - 50
        bar_height = bar_y_end - bar_y_start

        cv2.rectangle(frame, (bar_x - 15, bar_y_start),
                      (bar_x + 15, bar_y_end), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x - 15, bar_y_start),
                      (bar_x + 15, bar_y_end), (255, 255, 255), 2)

        # Target line
        target_y = bar_y_start + bar_height // 2
        cv2.line(frame, (bar_x - 20, target_y), (bar_x + 20, target_y),
                 (0, 255, 0), 2)

        # Current distance indicator
        # Map distance to bar position
        distance_range = target_distance * 2  # Show +/- target_distance
        normalized_distance = (current_z - target_distance) / distance_range
        current_y = int(target_y + normalized_distance * bar_height * 0.8)
        current_y = max(bar_y_start, min(bar_y_end, current_y))

        cv2.circle(frame, (bar_x, current_y), 8, (0, 165, 255), -1)

        # Labels
        cv2.putText(frame, "DIST", (bar_x - 20, bar_y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"{current_z * 100:.0f}cm", (bar_x - 30, current_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)