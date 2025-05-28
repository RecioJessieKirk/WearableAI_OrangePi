import torch
import cv2
import pyttsx3
import time
import numpy as np
import warnings
import subprocess
import sys
from pynput import keyboard

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def main():
    engine = pyttsx3.init()

    def speak(text):
        print(f"[TTS] {text}")
        engine.say(text)
        engine.runAndWait()

    def get_camera_index():
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.release()
                return index
        return None

    # === Set device with fallback ===
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[CUDA] Error: {e}")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    speak(f"Using device: {device}")

    try:
        model = torch.hub.load(
            './yolov5',
            'custom',
            path='yolomodels/yolov5n.pt',
            source='local',
            force_reload=True
        ).to(device)
        model.eval()
    except Exception as e:
        speak("Failed to load object detection model.")
        print(f"[ERROR] Model load failed: {e}")
        return

    c_pressed = False
    esc_pressed = False
    frame_to_analyze = None

    def on_press(key):
        nonlocal c_pressed, esc_pressed
        try:
            if key.char == 'c':
                c_pressed = True
        except AttributeError:
            if key == keyboard.Key.esc:
                esc_pressed = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    camera_index = get_camera_index()
    if camera_index is None:
        speak("No available camera found. Exiting.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        speak("Camera failed to open. Exiting.")
        return

    speak("Camera is ready. Press the Button to take a look.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                speak("Failed to grab frame.")
                break

            cv2.imshow("Live Camera - Press the Button", frame)

            if c_pressed:
                frame_to_analyze = frame.copy()
                break

            if esc_pressed:
                speak("Exiting.")
                break

            if cv2.getWindowProperty("Live Camera - Press the Button", cv2.WND_PROP_VISIBLE) < 1:
                speak("Window closed manually.")
                break

            cv2.waitKey(1)

    except KeyboardInterrupt:
        speak("Interrupted manually.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        listener.stop()

    if frame_to_analyze is not None:
        speak("Analyzing what I see...")

        h, w, _ = frame_to_analyze.shape
        center_x, center_y = w // 2, h // 2

        frame_rgb = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
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

        if closest_object:
            speak(f"I see a {closest_object} in front of you.")
        else:
            speak("I couldn't identify anything clearly.")

    # === Relaunch the NLP interface ===
    speak("Going back to listening mode.")
    subprocess.run([sys.executable, "integratedvoicenlp.py"])

if __name__ == "__main__":
    main()
