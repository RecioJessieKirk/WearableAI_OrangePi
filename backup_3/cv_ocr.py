import cv2
import pyttsx3
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

# === Suppress Warnings ===
warnings.filterwarnings("ignore", category=FutureWarning)

# === TTS Initialization ===
engine = pyttsx3.init()
engine.setProperty("rate", 175)

def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# === Whisper Tiny Setup ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === EasyOCR Reader Initialization ===
reader = easyocr.Reader(['en'], model_storage_directory='ocr_model', download_enabled=False)

# === Control Flags ===
trigger_read = threading.Event()
trigger_stop = threading.Event()

# === Voice Listener Thread ===
def listen_for_commands():
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("[VOICE] Listening for 'read' or 'stop'...")

    while not trigger_stop.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

        if len(frames) >= int(RATE / CHUNK * 1.2):
            audio_bytes = b''.join(frames)
            audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
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

# === Text Processing Function ===
def process_image(image):
    print("[OCR] Starting text detection...")
    results = reader.readtext(image)
    if not results:
        print("[OCR] No text detected")
        speak("No text detected.")
        return
    texts = [text for (_, text, _) in results]
    joined = '. '.join(texts) + '.'
    print(f"[OCR] Detected text: {joined}")
    speak(joined)

# === Main Application ===
def main():
    # === Start Voice Listener Thread
    voice_thread = threading.Thread(target=listen_for_commands)
    voice_thread.daemon = True
    voice_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not available.")
        speak("Camera not available.")
        return

    cv2.namedWindow("OCR Camera")
    speak("Camera is live. Say 'read' to read the text. Say 'stop' to go back.")

    while not trigger_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            speak("Failed to read camera.")
            break

        cv2.imshow("OCR Camera", frame)

        if trigger_read.is_set():
            trigger_read.clear()
            speak("Capturing and analyzing text.")
            captured = frame.copy()
            process_image(captured)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Going back to listening mode.")
    subprocess.run([sys.executable, "integratedvoicenlp.py"])

if __name__ == "__main__":
    main()
