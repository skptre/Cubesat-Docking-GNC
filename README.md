# CubeSat Universal Docking Interface

## Hardware Architecture
This software is designed to run on the primary vision processing node of the docking system:
* **Processor:** Raspberry Pi 5
* **Camera:** Arducam OV9281 (Monochrome Global Shutter)
* **Markers:** 1cm ArUco markers (DICT_4X4_50) arranged in a square pattern.

## How the Approach Logic Works (SNAP)
We use a Finite State Machine (FSM) to prevent the satellite from flying diagonal approach vectors, which is dangerous in space. The logic strictly forces lateral alignment before allowing any forward movement.

1. **STATION_KEEPING**: Stops momentum and holds the vehicle at the 1-meter Keep-Out-Zone (KOZ).
2. **ALIGNING_XY**: Fires lateral thrusters to center the X and Y axes. Forward thrust is locked.
3. **APPROACHING**: Once the lateral error is tight (<1cm), the system allows forward Z-axis thrust.
4. **SOFT_CAPTURE**: At 5cm out, camera distortion takes over. We drop the optical feed and trigger soft capture mechanism to take over and pull the mechanical interfaces together.

## Docking Parameters
* `MARKER_SIZE`: Currently set to `0.01` meters (1cm).
* `TARGET_DISTANCE`: The ideal station-keeping distance, currently set to `1.0` meters (100cm).
* `TERMINAL_DISTANCE`: `0.05` meters (5cm blind soft-capture zone)

## Usage
**Spin up the Flask server to stream the live GNC dashboard over your network:**
```bash
python web_docking.py
```
Then navigate to `http://<pi-ip-address>:5000` in your browser.

**To run the vision system with a local video feed:**
```bash
python main_docking_system.py
```

### Controls
While the camera window is active, use the following keyboard inputs:
* `q` - Quit the application and close the camera stream gracefully.
* `s` - Save a screenshot of the current frame (useful for logging alignment errors or testing).

