import numpy as np

class AlignmentCalculator:

    """
    Assumes 4 markers arranged in a square pattern on the target
    """

    def __init__(self, target_distance=0.3, required_marker_ids=[0, 1, 2, 3]):

        self.target_distance = target_distance # meters
        self.required_marker_ids = set(required_marker_ids)

    def calculate_target_center(self, ids, tvecs):
        if ids is None:
            return None

        detected_ids = set(ids.flatten())

        # Check for required markers
        if not self.required_marker_ids.issubset(detected_ids):
            missing = self.required_marker_ids - detected_ids
            print(f"Missing markers: {missing}")
            return None
        # Get positions of required markers
        positions = []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.required_marker_ids:
                positions.append(tvecs[i][0])

        # Calculate center:
        center = np.mean(positions, axis=0)
        return center

    def calculate_alignment(self, ids, tvecs):
        if ids is None:
            return None

        detected_ids = set(ids.flatten())

        # Check if we have all required markers
        if not self.required_marker_ids.issubset(detected_ids):
            missing = self.required_marker_ids - detected_ids
            print(f"Missing markers: {missing}")
            return None

        # Get position of required markers
        position =[]
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.required_marker_ids:
                position.append(tvecs[i][0])


        # Calculate center (average position)
        center = np.mean(position, axis=0)
        return center

    def calculate_alignment_error(self, center_position):

        if center_position is None:
            return None

        x, y, z = center_position

        # Ideal position: x=0, y=0 (centered), z=target_distance

        error_x = x # Positive = move camera right
        error_y = y # Positive = move camera down
        error_z = z - self.target_distance # Positive = too far

        lateral_error = np.sqrt(error_x**2 + error_y**2)
        total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)


        return {
            'error_x': error_x,
            'error_y': error_y,
            'error_z': error_z,
            'lateral_error': lateral_error,
            'total_error': total_error,
        }

    def get_movement_command(self, error_dict, threshold=0.01):

        if error_dict is None:
            return {'status': 'NO_MARKERS', 'command': 'Cannot detect all markers'}

        if error_dict['total_error'] < threshold:
            return {'status': 'ALIGNED', 'command': 'Perfect alignment'}

        commands = []

        # Lateral movement
        if abs(error_dict['error_x']) > threshold:
            direction = 'RIGHT' if error_dict['error_x'] > 0 else 'LEFT'
            commands.append(f"{direction} {abs(error_dict['error_x'])*100:.1f}cm")

        if abs(error_dict['error_y']) > threshold:
            direction = 'DOWN' if error_dict['error_y'] > 0 else 'UP'
            commands.append(f"{direction} {abs(error_dict['error_y'])*100:.1f}cm")

        # Distance
        if abs(error_dict['error_z']) > threshold:
            direction = 'FORWARD' if error_dict['error_z'] > 0 else 'BACKWARD'
            commands.append(f"{direction} {abs(error_dict['error_z'])*100:.1f}cm")

        return {
            'status': 'ALIGNING',
            'command': ', '.join(commands),
            'priority': commands[0] if commands else 'ALIGNED'
        }