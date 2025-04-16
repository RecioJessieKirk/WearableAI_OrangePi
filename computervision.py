from ultralytics import YOLO
import cv2
import pyttsx3
import time
import keyboard  # For non-blocking key detection
import numpy as np
import sys  # Allows exiting the script

# ✅ Function to Detect and Open First Available Camera
def get_camera_index():
    for index in range(5):  # Check first 5 indexes (0,1,2,3,4)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return None  # No available camera

# ✅ Try to Auto-Detect Camera
camera_index = get_camera_index()
if camera_index is None:
    print("No available camera found. Exiting...")
    exit()

# ✅ Initialize YOLO model
model = YOLO("yolomodels/yolov5n.pt")

# ✅ Initialize Text-to-Speech (TTS)
engine = pyttsx3.init()

# ✅ Track object announcements
last_announced_object = None
y_pressed_time = None  # Track when 'Y' is first pressed
y_hold_duration = 3  # Seconds to trigger announcement

try:
    # ✅ Start webcam and process detections
    for result in model.predict(source=camera_index, show=True, stream=True):
        frame = result.orig_img  # Get the original video frame
        h, w, _ = frame.shape  # Get frame dimensions
        center_x, center_y = w // 2, h // 2  # Calculate the center of the frame

        detected_objects = []
        closest_object = None
        min_distance = float('inf')  # Initialize with a large value

        # ✅ Process detected objects
        if result.boxes:
            for box in result.boxes:
                obj_name = model.names[int(box.cls)]  # Get object name
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2

                # Compute distance to frame center
                distance = np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2)

                # Update closest object
                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj_name

        # ✅ Detect if 'Y' is being held down
        if keyboard.is_pressed('y'):
            if y_pressed_time is None:
                y_pressed_time = time.time()  # Start timing when 'Y' is first pressed

            # Check if 'Y' is held for the required duration
            elif (time.time() - y_pressed_time) >= y_hold_duration:
                if closest_object:
                    if closest_object != last_announced_object:  # New object detected
                        print(f"Announcing object: {closest_object}")
                        engine.say(f"{closest_object}")
                    else:  # Same object detected
                        print(f"Still detecting: {closest_object}")
                        engine.say(f"Still detecting {closest_object}")

                    engine.runAndWait()
                    last_announced_object = closest_object  # Update last announced object

                y_pressed_time = None  # Reset timer after speaking
        else:
            y_pressed_time = None  # Reset if 'Y' is released

        # ✅ Detect if 'O' is pressed (Switch to OCR)
        if keyboard.is_pressed('o'):
            print("Switching to OCR mode...")
            cv2.destroyAllWindows()
            sys.exit(1)  # Exit the script to signal `main.py` to switch to OCR

        # ✅ Draw bounding boxes and display the frame
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                obj_name = model.names[int(box.cls)]
                color = (0, 255, 0) if obj_name == closest_object else (0, 0, 255)  # Highlight closest object in green

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, obj_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ✅ Draw a crosshair at the center of the screen
        cv2.drawMarker(frame, (center_x, center_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # ✅ Display the frame
        cv2.imshow("YOLO Detection", frame)

        # ✅ Ensure OpenCV window stays open
        if cv2.getWindowProperty("YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Stopping detection.")
            break

        # ✅ Allow exiting the loop with 'ESC'
        if keyboard.is_pressed('esc'):
            print("User pressed 'ESC'. Stopping detection.")
            break

        # ✅ Allow quitting with 'Q'
        if keyboard.is_pressed('q'):
            print("User pressed 'Q'. Stopping detection.")
            break

        cv2.waitKey(1)  # Ensures OpenCV window updates properly

except KeyboardInterrupt:
    print("Program interrupted manually.")

finally:
    cv2.destroyAllWindows()
    print("Detection stopped, and resources cleaned up.")
 