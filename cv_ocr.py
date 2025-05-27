import easyocr
import cv2
import pyttsx3
import warnings
import sys
import subprocess
import time
from pynput import keyboard

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize TTS engine
engine = pyttsx3.init()

def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# Initialize EasyOCR reader
reader = easyocr.Reader(
    ['en'],
    model_storage_directory='ocr_model',
    download_enabled=False
)

# Global flag set by pynput listener
capture_flag = False

def on_press(key):
    global capture_flag
    try:
        if key.char.lower() == 'c':
            print("[INPUT] 'C' pressed")  # DEBUG
            capture_flag = True
    except AttributeError:
        pass  # ignore special keys

def process_image(image):
    print("[OCR] Starting text detection")  # DEBUG
    results = reader.readtext(image)
    if not results:
        print("[OCR] No text detected")     # DEBUG
        speak("No text detected.")
        return

    texts = [text for (_, text, _) in results]
    joined = '. '.join(texts) + '.'
    print(f"[OCR] Detected combined text: {joined}")  # DEBUG
    speak(joined)

def main():
    global capture_flag

    # Start pynput listener in background
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("[INIT] Key listener started")   # DEBUG

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Camera not available.")
        speak("Camera not available.")
        return
    print("[INIT] Camera opened")          # DEBUG

    speak("Press the Button to capture and read text.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            speak("Failed to read frame.")
            break

        cv2.imshow("OCR - press the Button to capture", frame)

        # Poll the capture_flag
        if capture_flag:
            print("[FLOW] capture_flag detected!")  # DEBUG
            capture_flag = False

            # Freeze frame
            captured = frame.copy()
            cap.release()
            cv2.destroyAllWindows()
            print("[FLOW] Camera released and window closed")  # DEBUG

            speak("Analyzing captured text.")
            process_image(captured)

            speak("Going back to listening mode.")
            time.sleep(0.5)   # allow TTS flush

            listener.stop()
            print("[FLOW] Listener stopped, launching NLP script")  # DEBUG

            subprocess.Popen([sys.executable, "integratedvoicenlp.py"])
            return

        # Allow ESC to break if needed
        if cv2.waitKey(1) & 0xFF == 27:
            print("[INPUT] ESC pressed, exiting")  # DEBUG
            break

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()
    print("[CLEANUP] Exited main loop and cleaned up")  # DEBUG

if __name__ == "__main__":
    main()
