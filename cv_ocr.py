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
import gc

# === Manual GUI Mode Toggle ===
# Set to 0 for GUI mode (shows camera feed)
# Set to 1 for non-GUI/headless mode (no window)
mode = 1  # <=== CHANGE THIS TO 1 FOR NON-GUI

warnings.filterwarnings("ignore", category=FutureWarning)

# === TTS Setup ===
engine = pyttsx3.init()
engine.setProperty("rate", 175)

def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# === Whisper Setup ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === Control Flags ===
trigger_capture = threading.Event()
trigger_stop = threading.Event()

# === Voice Thread ===
def listen_for_commands():
    RATE = 16000
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    print("[VOICE] Listening for 'what is this' or 'stop'...")

    while not trigger_stop.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if len(frames) >= int(RATE / CHUNK * 1.5):
            audio_bytes = b"".join(frames)
            audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

            if np.abs(audio_np).mean() < 0.01:
                frames = []
                continue

            try:
                result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
                transcript = result.get("text", "").strip().lower()
                print(f"[VOICE] Heard: {transcript}")
                if "what is this" in transcript:
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

# === OCR Function ===
def read_text_from_image(image, reader):
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

# === Main Function ===
def main():
    # === Start Voice Thread First
    voice_thread = threading.Thread(target=listen_for_commands)
    voice_thread.daemon = True
    voice_thread.start()

    # === Camera Check
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if mode == 0:
        cv2.namedWindow("OCR Camera")

    # === EasyOCR Init (after camera is verified)
    try:
        reader = easyocr.Reader(['en'], model_storage_directory='ocr_model', download_enabled=False)
    except Exception as e:
        speak("Failed to load OCR model.")
        print(f"[OCR Error] {e}")
        return

    speak("Camera is live. Say 'what is this' to analyze. Say 'stop' to exit.")

    while not trigger_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            speak("Camera read failed.")
            break

        if mode == 0:
            cv2.imshow("OCR Camera", frame)

        if trigger_capture.is_set():
            trigger_capture.clear()
            speak("Analyzing text.")
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
    subprocess.run([sys.executable, "integratedvoicenlp.py"])

if __name__ == "__main__":
    main()
