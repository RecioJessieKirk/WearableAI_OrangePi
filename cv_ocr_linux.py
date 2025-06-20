import cv2
import easyocr
import time
import subprocess
import sys
import warnings
import whisper
import numpy as np
import pyaudio
import threading
import torch
import gc

from tts_handler import speak  # Your custom Linux TTS handler

# === Manual GUI Mode Toggle ===
# 0 = GUI mode (shows camera window)
# 1 = Headless mode (no GUI)
mode = 1

warnings.filterwarnings("ignore", category=FutureWarning)

# === Load Whisper Model ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === Control Flags ===
trigger_read = threading.Event()
trigger_stop = threading.Event()

def listen_for_commands():
    """Listen for voice commands 'read' or 'stop' using Whisper."""
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    print("[VOICE] Listening for 'read' or 'stop'...")

    while not trigger_stop.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        # Process audio every ~1.5 seconds
        if len(frames) >= int(RATE / CHUNK * 1.5):
            audio_bytes = b"".join(frames)
            audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

            # Skip if audio is mostly silence
            if np.abs(audio_np).mean() < 0.01:
                frames = []
                continue

            try:
                result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
                transcript = result.get("text", "").strip().lower()
                print(f"[VOICE] Heard: {transcript}")

                if "read" in transcript:
                    trigger_read.set()
                elif "stop" in transcript:
                    trigger_stop.set()
                    break

            except Exception as e:
                print(f"[Whisper error] {e}")

            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()

def read_text_from_image(image, reader):
    """Perform OCR on the image and speak detected text."""
    print("[OCR] Starting text detection...")
    results = reader.readtext(image)

    if not results:
        print("[OCR] No text detected.")
        speak("No text detected.")
        return

    texts = [text for (_, text, _) in results]
    joined = '. '.join(texts) + '.'
    print(f"[OCR] Detected: {joined}")
    speak(joined)

def main():
    voice_thread = threading.Thread(target=listen_for_commands)
    voice_thread.daemon = True
    voice_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if mode == 0:
        cv2.namedWindow("OCR Camera")

    try:
        reader = easyocr.Reader(['en'], model_storage_directory='ocr_model', download_enabled=False)
    except Exception as e:
        speak("Failed to load OCR model.")
        print(f"[OCR Error] {e}")
        return

    speak("Camera is live. Say 'read' to analyze. Say 'stop' to exit.")

    while not trigger_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            speak("Camera read failed.")
            break

        if mode == 0:
            cv2.imshow("OCR Camera", frame)

        if trigger_read.is_set():
            trigger_read.clear()
            speak("Reading text.")
            read_text_from_image(frame.copy(), reader)

        if mode == 0 and cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if mode == 0:
        cv2.destroyAllWindows()

    trigger_stop.set()
    del reader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speak("Going back to listening mode.")
    subprocess.run([sys.executable, "integratedvoicenlp_linux.py"])

if __name__ == "__main__":
    main()
