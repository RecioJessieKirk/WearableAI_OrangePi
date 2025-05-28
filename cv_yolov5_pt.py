import torch
import cv2
import time
import numpy as np
import pyttsx3
import warnings
import subprocess
import sys
import whisper
import pyaudio
import threading
import gc

# === MANUAL MODE SETTING ===
# 0 = GUI Mode (with camera window)
# 1 = No GUI Mode (for Orange Pi optimization)
manual_mode = 1

warnings.simplefilter("ignore", category=FutureWarning)

# === TTS SETUP ===
engine = pyttsx3.init()
engine.setProperty("rate", 175)

def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# === WHISPER LOAD ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === CONTROL FLAGS ===
trigger_detect = threading.Event()
trigger_stop = threading.Event()

# === VOICE THREAD ===
def listen_for_phrases():
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    print("[VOICE] Listening for 'what is it' or 'stop'...")

    while not trigger_stop.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if len(frames) >= int(RATE / CHUNK * 1.5):  # ~1.5s chunks
            audio_bytes = b"".join(frames)
            audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

            try:
                result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
                transcript = result.get("text", "").strip().lower()
                print(f"[VOICE] Heard: {transcript}")
                if "what is it" in transcript:
                    trigger_detect.set()
                elif "stop" in transcript:
                    trigger_stop.set()
                    break
            except Exception as e:
                print(f"[Whisper error] {e}")

            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()

# === MAIN FUNCTION ===
def main():
    # === DEVICE SETUP ===
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[CUDA Error] {e}")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    speak(f"Using device: {device}")

    # === LOAD YOLOv5 MODEL ===
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

    # === CAMERA SETUP ===
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return

    if manual_mode == 0:
        cv2.namedWindow("Object Detector")

    speak("Camera is live. Say 'what is it' to identify. Say 'stop' to go back.")

    # === START VOICE LISTENER THREAD ===
    voice_thread = threading.Thread(target=listen_for_phrases)
    voice_thread.daemon = True
    voice_thread.start()

    while not trigger_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            speak("Camera failed.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.pandas().xyxy[0]

        for _, det in detections.iterrows():
            label = det['name']
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            if manual_mode == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if manual_mode == 0:
            cv2.imshow("Object Detector", frame)

        if trigger_detect.is_set():
            trigger_detect.clear()
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            closest_obj = None
            min_dist = float('inf')

            for _, det in detections.iterrows():
                label = det['name']
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                obj_cx = (x1 + x2) // 2
                obj_cy = (y1 + y2) // 2

                dist = np.sqrt((obj_cx - center_x) ** 2 + (obj_cy - center_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = label

            if closest_obj:
                speak(f"I see a {closest_obj} in front of you.")
            else:
                speak("I couldn't identify anything clearly.")

        if manual_mode == 0:
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # === CLEANUP ===
    cap.release()
    if manual_mode == 0:
        cv2.destroyAllWindows()
    trigger_stop.set()

    # === CLEANUP RESOURCES ===
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speak("Going back to listening mode.")
    subprocess.run([sys.executable, "integratedvoicenlp.py"])

if __name__ == "__main__":
    main()
