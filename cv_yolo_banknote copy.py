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

    def extract_amount(label):
        try:
            number = int(label.lower().replace("peso", "").strip())
            return number
        except ValueError:
            return 0

    def compute_total(labels):
        return sum(extract_amount(label) for label in labels)

    def enhance_banknote_regions(image):
        # Convert to HSV for color-based masking
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_ranges = [
            ((5, 100, 100), (15, 255, 255)),     # Orange (20)
            ((0, 100, 100), (10, 255, 255)),     # Red (50)
            ((140, 100, 100), (160, 255, 255)),  # Purple (100)
            ((40, 50, 50), (85, 255, 255)),      # Green (200)
            ((20, 100, 100), (35, 255, 255)),    # Yellow (500)
            ((90, 100, 100), (130, 255, 255))    # Blue (1000)
        ]

        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            color_mask = cv2.bitwise_or(color_mask, mask)

        # === Edge detection mask ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # Combine color + edge mask
        combined_mask = cv2.bitwise_or(color_mask, edges_dilated)

        # Enhance only regions in mask
        enhanced = image.copy()
        enhanced[combined_mask > 0] = cv2.convertScaleAbs(enhanced[combined_mask > 0], alpha=1.5, beta=40)
        return enhanced

    def get_detections(frame):
        enhanced = enhance_banknote_regions(frame)
        frame_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections_df = results.pandas().xyxy[0]
        return detections_df['name'].tolist()

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
            path='yolomodels/best_bank.pt',
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

    speak("Camera is ready. Press the Button to scan the money.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                speak("Failed to grab frame.")
                break

            cv2.imshow("Banknote Detector - Press the Button", frame)

            if c_pressed:
                frame_to_analyze = frame.copy()
                break

            if esc_pressed:
                speak("Exiting.")
                break

            if cv2.getWindowProperty("Banknote Detector - Press the Button", cv2.WND_PROP_VISIBLE) < 1:
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
        speak("Analyzing the banknotes...")

        first_detected = get_detections(frame_to_analyze)
        first_set = set(first_detected)

        if not first_set:
            speak("I couldn't recognize any banknotes.")
        else:
            speak("Scanning again to be sure.")
            time.sleep(1)

            second_detected = get_detections(frame_to_analyze)
            second_set = set(second_detected)

            all_detected = list(first_set.union(second_set))
            print("[INFO] All detected:", all_detected)

            total_amount = compute_total(all_detected)

            if total_amount > 0:
                speak(f"Total money is {total_amount} pesos.")
            else:
                speak("I couldn't recognize any banknotes.")

    speak("Going back to listening mode.")
    subprocess.run([sys.executable, "integratedvoicenlp.py"])

if __name__ == "__main__":
    main()
