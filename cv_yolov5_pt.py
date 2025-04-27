import torch
import cv2
import pyttsx3
import time
import numpy as np
import sys
import warnings
from pynput import keyboard  # ✅ New: using pynput

# Suppress warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def main():
    # ✅ Initialize TTS engine
    engine = pyttsx3.init()

    # ✅ Function to speak and print
    def speak(text):
        print(text)
        engine.say(text)
        engine.runAndWait()

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
    speak(f"Using device: {device}")
    model = torch.hub.load(
        './yolov5',
        'custom',
        path=r'yolomodels/yolov5n.pt',
        source='local',
        force_reload=True
    ).to(device)
    model.eval()

    # ✅ Initialize state tracking
    last_announced_object = None
    y_pressed = False
    c_pressed = False
    esc_pressed = False

    y_pressed_time = None
    c_pressed_time = None
    y_hold_duration = 3  # seconds
    c_hold_duration = 3.5  # seconds

    # ✅ Define key press and release handlers
    def on_press(key):
        nonlocal y_pressed, c_pressed, esc_pressed, y_pressed_time, c_pressed_time

        try:
            if key.char == 'y':
                if not y_pressed:
                    y_pressed = True
                    y_pressed_time = time.time()
            if key.char == 'c':
                if not c_pressed:
                    c_pressed = True
                    c_pressed_time = time.time()
        except AttributeError:
            if key == keyboard.Key.esc:
                esc_pressed = True

    def on_release(key):
        nonlocal y_pressed, c_pressed, y_pressed_time, c_pressed_time

        try:
            if key.char == 'y':
                y_pressed = False
                y_pressed_time = None
            if key.char == 'c':
                c_pressed = False
                c_pressed_time = None
        except AttributeError:
            pass

    # ✅ Set up pynput listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # ✅ Get camera
    camera_index = get_camera_index()
    if camera_index is None:
        speak("No available camera found. Exiting.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        speak("Camera failed to open. Exiting.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                speak("Failed to grab frame.")
                break

            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            detections = results.pandas().xyxy[0]

            closest_object = None
            min_distance = float('inf')

            for _, det in detections.iterrows():
                obj_name = det['name']
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2

                distance = np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj_name

            # ✅ Handle Y hold
            if y_pressed and y_pressed_time:
                if (time.time() - y_pressed_time) >= y_hold_duration:
                    if closest_object:
                        if closest_object != last_announced_object:
                            speak(f"Announcing {closest_object}")
                        else:
                            speak(f"Still detecting {closest_object}")
                        last_announced_object = closest_object
                    y_pressed_time = None  # Reset so it only announces once per hold

            # ✅ Handle C hold
            if c_pressed and c_pressed_time:
                if (time.time() - c_pressed_time) >= c_hold_duration:
                    speak("Switching back to NLP.")
                    break

            # ✅ Draw Detections
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                obj_name = det['name']
                color = (0, 255, 0) if obj_name == closest_object else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, obj_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.drawMarker(frame, (center_x, center_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            cv2.imshow("YOLOv5 Detection", frame)

            if cv2.getWindowProperty("YOLOv5 Detection", cv2.WND_PROP_VISIBLE) < 1:
                speak("Window closed.")
                break

            if esc_pressed:
                speak("ESC pressed. Exiting.")
                break

            cv2.waitKey(1)

    except KeyboardInterrupt:
        speak("Interrupted manually.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        listener.stop()  # ✅ Stop the pynput listener
        speak("Resources cleaned up.")

if __name__ == "__main__":
    main()
