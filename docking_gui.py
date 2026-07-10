import cv2
import numpy as np


class DockingGUI:

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.48
    THICK = 1

    # State -> accent color (BGR)
    STATE_COLORS = {
        'STATION_KEEPING': (80, 220, 80),     # green: holding, aligned
        'APPROACHING':     (255, 210, 60),    # cyan-ish: closing in
        'ALIGNING_XY':     (60, 170, 255),    # orange: correcting lateral
        'BACKING_UP':      (60, 170, 255),    # orange: correcting range
        'SOFT_CAPTURE':    (255, 80, 255),    # magenta: magnets engaged
    }
    COLOR_BAD    = (70, 70, 255)              # red: no markers / no pose
    COLOR_TEXT   = (235, 235, 235)
    COLOR_DIM    = (170, 170, 170)
    COLOR_TARGET = (255, 0, 255)
    COLOR_ACCENT = (60, 170, 255)

    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _strip(self, frame, x, y, w, h, darken=0.35):
        """Darken a rectangle of the frame in place (translucent backdrop)."""
        x2, y2 = min(x + w, self.width), min(y + h, self.height)
        x, y = max(x, 0), max(y, 0)
        roi = frame[y:y2, x:x2]
        roi[:] = (roi * darken).astype(np.uint8)

    def _text(self, frame, s, org, color=None, scale=None):
        cv2.putText(frame, s, org, self.FONT, scale or self.SCALE,
                    color or self.COLOR_TEXT, self.THICK, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # elements
    # ------------------------------------------------------------------

    def draw_crosshair(self, frame):
        c = (255, 255, 255)
        cv2.line(frame, (self.center_x - 10, self.center_y),
                 (self.center_x + 10, self.center_y), c, 1, cv2.LINE_AA)
        cv2.line(frame, (self.center_x, self.center_y - 10),
                 (self.center_x, self.center_y + 10), c, 1, cv2.LINE_AA)

    def draw_target_point(self, frame, center_position, camera_matrix):
        """Project the CAMERA-FRAME target position into the image."""
        if center_position is None:
            return None
        center_3d = np.asarray(center_position, dtype=np.float64).reshape(1, 1, 3)
        center_2d, _ = cv2.projectPoints(center_3d, np.zeros((3, 1)),
                                         np.zeros((3, 1)), camera_matrix,
                                         np.zeros((5, 1)))
        tx, ty = int(center_2d[0][0][0]), int(center_2d[0][0][1])
        cv2.circle(frame, (tx, ty), 14, self.COLOR_TARGET, 2, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), 2, self.COLOR_TARGET, -1, cv2.LINE_AA)
        return tx, ty

    def draw_alignment_arrows(self, frame, error_dict, target_point):
        if error_dict is None or target_point is None:
            return
        tx, ty = target_point
        if abs(error_dict['error_x']) > 0.01:
            cv2.arrowedLine(frame, (self.center_x, self.center_y),
                            (tx, self.center_y), self.COLOR_ACCENT, 2,
                            cv2.LINE_AA, tipLength=0.25)
        if abs(error_dict['error_y']) > 0.01:
            cv2.arrowedLine(frame, (self.center_x, self.center_y),
                            (self.center_x, ty), self.COLOR_ACCENT, 2,
                            cv2.LINE_AA, tipLength=0.25)

    def draw_status_panel(self, frame, command_dict, error_dict):
        if command_dict is None:
            return
        status = command_dict['status']
        color = self.STATE_COLORS.get(status, self.COLOR_BAD)

        h = 92 if error_dict is not None else 54
        self._strip(frame, 10, 10, 320, h)

        # state dot + name
        cv2.circle(frame, (26, 30), 6, color, -1, cv2.LINE_AA)
        self._text(frame, status, (40, 35), color)
        # command
        self._text(frame, command_dict['command'], (22, 58))

        if error_dict is not None:
            e = error_dict
            self._text(frame,
                       f"err  X {e['error_x']*100:+.1f}  "
                       f"Y {e['error_y']*100:+.1f}  "
                       f"Z {e['error_z']*100:+.1f} cm",
                       (22, 82), self.COLOR_DIM)

    def draw_pose_strip(self, frame, range_m, yaw, pitch, roll, n_used, n_total=4):
        # Bottom strip: measured range, relative attitude, markers in use.
        y = self.height - 34
        self._strip(frame, 10, y, self.width - 96, 26)
        self._text(frame,
                   f"range {range_m*100:6.1f} cm   "
                   f"yaw {yaw:+5.1f}  pitch {pitch:+5.1f}  roll {roll:+5.1f}   "
                   f"markers {n_used}/{n_total}",
                   (22, y + 18))

    def draw_distance_indicator(self, frame, center_position, target_distance, scale_max=None):
        if center_position is None:
            return
        z = float(center_position[2])
        full = scale_max if scale_max is not None else 2.0 * target_distance                # top of scale

        bx = self.width - 40
        top, bot = 70, self.height - 46
        bh = bot - top

        self._strip(frame, bx - 26, top - 34, 60, bh + 68)
        cv2.rectangle(frame, (bx - 7, top), (bx + 7, bot),
                      (120, 120, 120), 1, cv2.LINE_AA)

        self._text(frame, "RANGE", (bx - 24, top - 14), self.COLOR_DIM, 0.4)
        self._text(frame, f"{full*100:.0f}", (bx + 12, top + 5), self.COLOR_DIM, 0.38)
        self._text(frame, "0", (bx + 12, bot + 4), self.COLOR_DIM, 0.38)

        # setpoint tick (green)
        sy = int(bot - (target_distance / full) * bh)
        cv2.line(frame, (bx - 12, sy), (bx + 12, sy), (80, 220, 80), 2, cv2.LINE_AA)

        # current range dot + value
        cy = int(bot - (min(max(z, 0.0), full) / full) * bh)
        cv2.circle(frame, (bx, cy), 6, self.COLOR_ACCENT, -1, cv2.LINE_AA)
        label = f"{z*100:.1f}"
        ty = min(max(cy + 5, top + 12), bot - 4)
        self._text(frame, label, (bx - 24 - len(label) * 2, ty),
                   self.COLOR_ACCENT, 0.42)