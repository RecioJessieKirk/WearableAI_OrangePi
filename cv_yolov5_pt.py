import torch
import cv2
import pyttsx3
import time
import keyboard
import numpy as np
import sys
import warnings

# Suppress warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def main():
    # ✅ Function to Detect and Open First Available Camera
    def get_camera_index():
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.release()
                return index
        return None

    # ✅ Load YOLOv5 model using PyTorch Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = torch.hub.load(
        './yolov5',
        'custom',
        path=r'yolomodels/yolov5n.pt',
        source='local',
        force_reload=True
    ).to(device)
    model.eval()

    # ✅ Initialize TTS engine
    engine = pyttsx3.init()

    # ✅ Track object announcements
    last_announced_object = None
    y_pressed_time = None
    c_pressed_time = None
    y_hold_duration = 3
    c_hold_duration = 3.5

    # ✅ Get available camera
    camera_index = get_camera_index()
    if camera_index is None:
        print("No available camera found. Exiting...")
        return  # Use return instead of sys.exit()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Camera failed to open. Exiting...")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2

            # ✅ Convert frame for YOLOv5
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            detections = results.pandas().xyxy[0]

            closest_object = None
            min_distance = float('inf')

            # ✅ Process detections
            for _, det in detections.iterrows():
                obj_name = det['name']
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2

                distance = np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj_name

            # ✅ Detect if 'Y' is held
            if keyboard.is_pressed('y'):
                if y_pressed_time is None:
                    y_pressed_time = time.time()
                elif (time.time() - y_pressed_time) >= y_hold_duration:
                    if closest_object:
                        if closest_object != last_announced_object:
                            print(f"Announcing: {closest_object}")
                            engine.say(f"{closest_object}")
                        else:
                            print(f"Still detecting: {closest_object}")
                            engine.say(f"Still detecting {closest_object}")
                        engine.runAndWait()
                        last_announced_object = closest_object
                    y_pressed_time = None
            else:
                y_pressed_time = None

            # ✅ Detect if 'C' is held (new logic!)
            if keyboard.is_pressed('c'):
                if c_pressed_time is None:
                    c_pressed_time = time.time()
                elif (time.time() - c_pressed_time) >= c_hold_duration:
                    print("Switching back to NLP...")
                    break  # Exit gracefully
            else:
                c_pressed_time = None

            # ✅ Draw detections
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                obj_name = det['name']
                color = (0, 255, 0) if obj_name == closest_object else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, obj_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ✅ Crosshair
            cv2.drawMarker(frame, (center_x, center_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # ✅ Display frame
            cv2.imshow("YOLOv5 Detection", frame)

            if cv2.getWindowProperty("YOLOv5 Detection", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed.")
                break

            if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
                print("Exiting...")
                break

            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Interrupted manually.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up.")

if __name__ == "__main__":
    main()
