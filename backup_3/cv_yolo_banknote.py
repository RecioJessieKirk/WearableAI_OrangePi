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

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# === TTS SETUP ===
engine = pyttsx3.init()
engine.setProperty("rate", 175)

def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# === Whisper Tiny LOAD ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === Global Flag for Capture ===
trigger_capture = threading.Event()

def listen_for_yes():
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("[VOICE] Listening in real-time for 'yes'...")

    frames = []

    while not trigger_capture.is_set():
        data = stream.read(CHUNK)
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
                    break
            except Exception as e:
                print(f"[Whisper error] {e}")

            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()

# === Detection Utils ===
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

# === MAIN FUNCTION ===
def main():
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        except:
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
        print(f"[ERROR] {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return

    speak("Camera is ready. Say 'yes' when you're ready.")
    cv2.namedWindow("Banknote Scanner")

    # Start voice listener in background
    voice_thread = threading.Thread(target=listen_for_yes)
    voice_thread.daemon = True
    voice_thread.start()

    frame_to_analyze = None

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Failed to read from camera.")
            break

        cv2.imshow("Banknote Scanner", frame)

        if trigger_capture.is_set():
            frame_to_analyze = frame.copy()
            break

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_to_analyze is None:
        speak("No image captured.")
        return

    # === Detect & Sum ===
    speak("Analyzing the money...")
    first = get_detections(frame_to_analyze, model)
    time.sleep(1)
    second = get_detections(frame_to_analyze, model)
    all_labels = list(set(first + second))

    print(f"[INFO] Detected: {all_labels}")
    total = compute_total(all_labels)

    if total > 0:
        speak(f"Total money is {total} pesos.")
    else:
        speak("I couldn't recognize any banknotes.")

    speak("Going back to listening mode.")
    subprocess.run([sys.executable, "integratedvoicenlp.py"])

if __name__ == "__main__":
    main()
