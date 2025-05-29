import torch
import cv2
import time
import numpy as np
import warnings
import subprocess
import sys
import pyaudio
import whisper
import threading
import gc
from queue import Queue

# === MODE TOGGLE ===
manual_mode = 1  # 0 = GUI (OpenCV windows), 1 = No GUI (headless mode for Orange Pi)

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# === TTS Handler ===
def tts_handler(text):
    print(f"[TTS] {text}")
    subprocess.run(["espeak", "-s", "175", text])

# === Whisper Tiny LOAD ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === Global Flag for Capture ===
trigger_capture = threading.Event()

# === Voice Listening Thread ===
def listen_for_yes(tts_func, trigger_event):
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        tts_func("Audio input not available.")
        print(f"[VOICE ERROR] {e}")
        return

    tts_func("Listening for confirmation word 'yes'.")
    frames = []

    while not trigger_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            # 1.2 seconds buffer for detection
            if len(frames) >= int(RATE / CHUNK * 1.2):
                audio_bytes = b"".join(frames)
                audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
                try:
                    result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
                    transcript = result.get("text", "").strip().lower()
                    print(f"[VOICE] Heard: {transcript}")
                    if "yes" in transcript:
                        trigger_event.set()
                        break
                except Exception as e:
                    print(f"[Whisper error] {e}")
                frames = []
        except Exception as e:
            print(f"[Audio stream error] {e}")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

# === Detection Utils ===
def extract_amount(label):
    try:
        # Expects labels like "20 peso" or "100 peso"
        return int(''.join(filter(str.isdigit, label)))
    except Exception:
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

# === Main Function ===
def main():
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    tts_handler(f"Using device: {device}")

    # Load YOLOv5 model
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
        tts_handler("Failed to load object detection model.")
        print(f"[ERROR] {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tts_handler("Camera not available.")
        return

    tts_handler("Camera is ready. Please say 'yes' when you are ready.")

    if manual_mode == 0:
        cv2.namedWindow("Banknote Scanner")

    voice_thread = threading.Thread(target=listen_for_yes, args=(tts_handler, trigger_capture))
    voice_thread.daemon = True
    voice_thread.start()

    frame_to_analyze = None

    while True:
        ret, frame = cap.read()
        if not ret:
            tts_handler("Failed to read from camera.")
            break

        if manual_mode == 0:
            cv2.imshow("Banknote Scanner", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        if trigger_capture.is_set():
            frame_to_analyze = frame.copy()
            break

        if manual_mode == 1:
            time.sleep(0.05)

    cap.release()
    if manual_mode == 0:
        cv2.destroyAllWindows()

    if frame_to_analyze is None:
        tts_handler("No image captured.")
        return

    tts_handler("Analyzing the money, please wait.")
    first_detections = get_detections(frame_to_analyze, model)
    time.sleep(1)
    second_detections = get_detections(frame_to_analyze, model)
    all_labels = list(set(first_detections + second_detections))

    print(f"[INFO] Detected labels: {all_labels}")
    total_amount = compute_total(all_labels)

    if total_amount > 0:
        tts_handler(f"The total money detected is {total_amount} pesos.")
    else:
        tts_handler("I couldn't recognize any banknotes.")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tts_handler("Going back to listening mode.")
    subprocess.run([sys.executable, "integratedvoicenlp_linux.py"])

if __name__ == "__main__":
    main()
