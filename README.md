# CubeSat Universal Docking Interface

## Hardware Architecture
This software is designed to run on the primary vision processing node of the docking system:
* **Processor:** Raspberry Pi 5
* **Camera:** Arducam OV9281 (Monochrome Global Shutter)
* **Markers:** 4x ArUco, `DICT_4X4_50`, IDs 0–3, 2 cm, one near each corner of the 10x10 cm face

## How Pose Estimation Works
The four markers are treated as a single rigid **constellation** (`cv2.aruco.Board`). Their true positions on the face are measured once and recorded in `face_model.py` (`MARKER_CENTERS`), with the origin defined at the docking-axis center. Each frame, every detected marker corner is matched against that map and one `solvePnP` call recovers the full 6-DOF pose of the face:
 
* **Position** — vector from the camera to the docking-axis center, then shifted
  by the camera's mounting offset (`CAMERA_IN_INTERFACE_FRAME`) so errors are
  expressed interface-to-interface, not lens-to-target.
* **Attitude** — relative yaw / pitch / roll of the target face, displayed live.
* Tracking degrades gracefully: any subset of visible markers works (1–4).

## How the Approach Logic Works (bench FSM)
A finite state machine turns the measured error into movement commands:
 
1. **ALIGNING_XY** — lateral error > 1 cm: command lateral correction, hold Z.
2. **APPROACHING / BACKING_UP** — laterally aligned: command Z toward the setpoint.
3. **STATION_KEEPING** — holding at `TARGET_DISTANCE` within tolerance.
4. **SOFT_CAPTURE** — < 5 cm: optics blackout, magnetic capture takes over.

Note: this axis-sequenced FSM is a bench harness; the flight architecture gates mission phases, not axes.

## Key Constants
| Constant | Where | Value |
|---|---|---|
| `MARKER_SIZE` | `face_model.py` | 0.020 m |
| `MARKER_CENTERS` | `face_model.py` | measured per build (**placeholder square until calipered/CAD**) |
| `CAMERA_IN_INTERFACE_FRAME` | `face_model.py` | (-0.0475, 0, 0) m (change depending on camera offset) |
| `TARGET_DISTANCE` | `web_docking.py` | 1.0 m (station-keeping setpoint) |
| Camera intrinsics | `camera_calibration.py` | **estimated (focal = 0.7 x width) — checkerboard calibration pending** |

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

### Pogo-pin data-rate benchmark (two mated modules)
```bash
python3 benchmark_transfer.py recv   # module A
python3 benchmark_transfer.py send   # module B
```
Sweeps payload sizes 16–4096 B (50 packets each) and reports goodput, efficiency against the 11,520 B/s physical ceiling (115200 baud, 8N1), mean round-trip time, retries, and failures. Saves `benchmark_results.json` for plotting throughput vs packet size.

### Reading the HUD
* Top-left: FSM state (colored dot), current command, signed X/Y/Z errors (cm)
* Bottom: measured range, relative yaw/pitch/roll (deg), markers in use
* Right bar: measured range on a fixed scale; green tick = setpoint, dot = current

## Known Limitations
* Camera intrinsics are estimated; ranges read a few percent off until
  checkerboard calibration is done.
* `MARKER_CENTERS` holds placeholder square values until the as-built
  positions are measured (calipers or CAD).
* FSM is bench-only (axis-sequenced); flight version gates phases.
* Data-rate benchmark measures the link while mated and static; rates under
  vibration or partial contact during capture are untested.

