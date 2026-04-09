# CubeSat Universal Docking Interface

## Hardware Architecture
This software is designed to run on the primary vision processing node of the docking system:
* **Processor:** Raspberry Pi 5
* **Camera:** Arducam OV9281 (Monochrome Global Shutter)
* **Markers:** 5cm ArUco markers (DICT_4X4_50) arranged in a square pattern.

## Usage
To launch the full docking vision system, run:
```bash
python main_docking_system.py
```

### Controls
While the camera window is active, use the following keyboard inputs:
* `q` - Quit the application and close the camera stream gracefully.
* `s` - Save a screenshot of the current frame (useful for logging alignment errors or testing).

## Docking Parameters
* `MARKER_SIZE`: Currently set to `0.05` meters (5cm).
* `TARGET_DISTANCE`: The ideal station-keeping distance, currently set to `0.3` meters (30cm).
