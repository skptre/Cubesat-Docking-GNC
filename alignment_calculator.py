import numpy as np

class AlignmentCalculator:
    """

    Finite State Machine (SNAP Logic):
    - NO_MARKERS: Camera cannot detect all required ArUco markers.
    - ACQUIRING: System initialization state.
    - ALIGNING_XY: Forward thrust halted. Firing lateral thrusters to center X/Y axes.
    - APPROACHING: X/Y axes are within strict tolerance. Firing forward (Z) thrusters.
    - BACKING_UP: Satellite has drifted too close but is not perfectly aligned.
    - STATION_KEEPING: Perfectly aligned at the safe 1m holding distance.
    - SOFT_CAPTURE: Distance is <5cm. GNC disabled. Magnetic soft capture engages.
    """
    def __init__(self, target_distance=0.3):
        self.target_distance = target_distance # 1.0 meters
        
        #FSM State Tracking & Tolerances
        self.current_state = 'ACQUIRING'
        self.xy_tolerance = 0.01  # 1 cm lateral tolerance
        self.z_tolerance = 0.02   # 2 cm distance tolerance
        self.terminal_distance = 0.05 # 5cm blind coast distance

    
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

    def get_movement_command(self, error_dict):
        if error_dict is None:
            self.current_state = 'NO_MARKERS'
            return {'status': self.current_state, 'command': 'Cannot detect all markers'}

        err_x = error_dict['error_x']
        err_y = error_dict['error_y']
        err_z = error_dict['error_z'] 
        lat_error = error_dict['lateral_error']
        
        current_z_distance = err_z + self.target_distance

        # 1. TERMINAL PHASE OVERRIDE
        if current_z_distance <= self.terminal_distance:
            self.current_state = 'SOFT_CAPTURE'
            return {
                'status': self.current_state, 
                'command': 'OPTICS DISABLED - GNC BLACKOUT'
            }

        # 2. ALIGN LATERAL (X/Y) FIRST
        if lat_error > self.xy_tolerance:
            self.current_state = 'ALIGNING_XY'
            commands = []
            
            if abs(err_x) > self.xy_tolerance:
                direction = 'RIGHT' if err_x > 0 else 'LEFT'
                commands.append(f"{direction} {abs(err_x)*100:.1f}cm")
                
            if abs(err_y) > self.xy_tolerance:
                direction = 'DOWN' if err_y > 0 else 'UP'
                commands.append(f"{direction} {abs(err_y)*100:.1f}cm")
                
            return {
                'status': self.current_state,
                'command': ' + '.join(commands)
            }

        # 3. ALIGN DISTANCE (Z)
        if abs(err_z) > self.z_tolerance:
            self.current_state = 'APPROACHING' if err_z > 0 else 'BACKING_UP'
            direction = 'FORWARD' if err_z > 0 else 'REVERSE'
            
            return {
                'status': self.current_state,
                'command': f"{direction} {abs(err_z)*100:.1f}cm"
            }

        # 4. PERFECTLY ALIGNED
        self.current_state = 'STATION_KEEPING'
        return {
            'status': self.current_state, 
            'command': 'HOLD POSITION - READY'
        }