import cv2
import torch
import warnings
import time
import subprocess
import sys
import threading
import pyaudio
import numpy as np
import whisper

from tts_handler import speak  # Import speak function directly

# === Suppress Warnings ===
warnings.simplefilter("ignore", category=FutureWarning)

# === Manual Mode Setting ===
# 0 = GUI Mode (with camera window)
# 1 = No GUI Mode (headless, e.g. Orange Pi)
manual_mode = 1  # Change to 0 to enable GUI

# === Whisper setup ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === Voice trigger flag ===
trigger_stop = threading.Event()

def listen_for_stop():
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("[VOICE] Listening for 'go back'...")

    while not trigger_stop.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

        if len(frames) >= int(RATE / CHUNK * 1.2):
            audio_bytes = b"".join(frames)
            audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
            try:
                result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
                transcript = result.get("text", "").strip().lower()
                print(f"[VOICE] Heard: {transcript}")
                if "go back" in transcript:
                    trigger_stop.set()
                    break
            except Exception as e:
                print(f"[Whisper error] {e}")
            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()

# === Device & model setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speak(f"Using device: {device}")

model = torch.hub.load(
    './yolov5',
    'custom',
    path='yolomodels/best_pothole.pt',
    source='local',
    force_reload=True
).to(device)
model.eval()
model.conf = 0.4

# === Distance Estimation Parameters ===
FOCAL_LENGTH = 700
ASSUMED_OBJECT_WIDTH_CM = 30
THRESHOLD_DISTANCE_CM = 40
ANNOUNCE_DELAY = 5  # seconds

def estimate_distance(width_px):
    if width_px == 0:
        return float('inf')
    return (ASSUMED_OBJECT_WIDTH_CM * FOCAL_LENGTH) / width_px

# === Open webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    speak("Camera not available.")
    sys.exit(1)

if manual_mode == 0:
    cv2.namedWindow("Pothole Detection")

speak("Pothole detection started. Say 'go back' when you're done.")

# === Start voice thread ===
voice_thread = threading.Thread(target=listen_for_stop)
voice_thread.daemon = True
voice_thread.start()

last_announce_time = 0

# === Main loop ===
while not trigger_stop.is_set():
    ret, frame = cap.read()
    if not ret:
        speak("Failed to read frame.")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    margin = int(frame_width * 0.1)

    results = model(frame)
    detections = results.pandas().xyxy[0]

    announced = False
    current_time = time.time()

    for _, row in detections.iterrows():
        label = row['name']
        conf = row['confidence']
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        width_px = x2 - x1
        object_center_x = (x1 + x2) // 2
        distance = estimate_distance(width_px)

        # Determine position
        if object_center_x < frame_center_x - margin:
            side = "left"
        elif object_center_x > frame_center_x + margin:
            side = "right"
        else:
            side = "in the middle"

        # Draw on frame if GUI mode
        if manual_mode == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} - {int(distance)}cm ({side})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Announce pothole if close and time passed
        if not announced and distance <= THRESHOLD_DISTANCE_CM and (current_time - last_announce_time) > ANNOUNCE_DELAY:
            speak(f"{label} detected at {int(distance)} centimeters {side}")
            last_announce_time = current_time
            announced = True

    if manual_mode == 0:
        cv2.imshow("Pothole Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit GUI mode early
            break

# === Cleanup ===
print("Cleaning up...")
cap.release()
if manual_mode == 0:
    cv2.destroyAllWindows()

speak("Going back to listening mode..")
subprocess.run([sys.executable, "integratedvoicenlp.py"])
