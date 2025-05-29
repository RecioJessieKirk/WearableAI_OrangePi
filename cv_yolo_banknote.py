import torch
import cv2
import pyttsx3
import time
import numpy as np
import warnings
import subprocess
import sys
import pyaudio
import whisper
import threading
import gc

manual_mode = 1  # No GUI mode

warnings.simplefilter("ignore", category=FutureWarning)

engine = pyttsx3.init()
engine.setProperty("rate", 175)
def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

trigger_capture = threading.Event()
trigger_stop = threading.Event()  # New flag for exit

def listen_for_yes_stop():
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    print("[VOICE] Listening for 'yes' to capture or 'stop' to exit")

    while not trigger_stop.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if len(frames) >= int(RATE / CHUNK * 1.2):
            audio_bytes = b"".join(frames)
            audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
            try:
                result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
                transcript = result.get("text", "").strip().lower()
                print(f"[VOICE] Heard: {transcript}")
                if "yes" in transcript:
                    trigger_capture.set()
                elif "stop" in transcript:
                    trigger_stop.set()
                    break
            except Exception as e:
                print(f"[Whisper error] {e}")

            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()

def extract_amount(label):
    try:
        return int(label.lower().replace("peso", "").strip())
    except:
        return 0

def compute_total(labels):
    return sum(extract_amount(label) for label in labels)

def enhance_banknote_regions(image):
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
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(color_mask, edges_dilated)
    enhanced = image.copy()
    enhanced[combined_mask > 0] = cv2.convertScaleAbs(enhanced[combined_mask > 0], alpha=1.5, beta=40)
    return enhanced

def get_detections(frame, model):
    enhanced = enhance_banknote_regions(frame)
    frame_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    return results.pandas().xyxy[0]['name'].tolist()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
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
        print(f"[ERROR] {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return

    if manual_mode == 0:
        cv2.namedWindow("Banknote Scanner")

    speak("Camera ready. Say 'yes' to capture or 'stop' to exit.")

    voice_thread = threading.Thread(target=listen_for_yes_stop)
    voice_thread.daemon = True
    voice_thread.start()

    while not trigger_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            speak("Failed to read from camera.")
            break

        if manual_mode == 0:
            cv2.imshow("Banknote Scanner", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if trigger_capture.is_set():
            trigger_capture.clear()
            speak("Analyzing the money...")
            first = get_detections(frame, model)
            time.sleep(1)
            second = get_detections(frame, model)
            all_labels = list(set(first + second))
            print(f"[INFO] Detected: {all_labels}")
            total = compute_total(all_labels)
            if total > 0:
                speak(f"Total money is {total} pesos.")
            else:
                speak("I couldn't recognize any banknotes.")
            speak("Say 'yes' to capture again or 'stop' to exit.")

        if manual_mode == 1:
            time.sleep(0.05)  # light sleep for CPU friendly loop

    cap.release()
    if manual_mode == 0:
        cv2.destroyAllWindows()
    trigger_stop.set()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speak("Exiting banknote scanner.")
    print("[INFO] Returning to integratedvoicenlp.py")

    try:
        subprocess.run([sys.executable, "integratedvoicenlp.py"])
    except Exception as e:
        print(f"[ERROR] Failed to restart integratedvoicenlp.py: {e}")

if __name__ == "__main__":
    main()
