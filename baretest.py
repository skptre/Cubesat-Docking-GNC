from picamera2 import Picamera2
import time
import cv2

print("--- BARE MINIMUM PICAMERA2 TEST ---")
try:
    picam2 = Picamera2()
    
    config = picam2.create_preview_configuration(main={"size": (640, 400)})
    picam2.configure(config)
    picam2.start()
    
    print("Camera started. Waiting 3 seconds for hardware to stabilize...")
    time.sleep(3)
    
    print("Attempting to grab a single frame...")
    frame = picam2.capture_array()
    
    print(f"SUCCESS! Frame grabbed. Shape: {frame.shape}")
    cv2.imwrite("test_frame.jpg", frame)
    print("Saved image to test_frame.jpg")

except Exception as e:
    print(f"TEST FAILED: {e}")
finally:
    try:
        picam2.stop()
        print("Camera hardware safely released.")
    except:
        pass