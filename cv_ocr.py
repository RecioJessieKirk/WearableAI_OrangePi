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
    print("[TTS] done")

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
            print("[INPUT] 'C' pressed")
            capture_flag = True
    except AttributeError:
        pass

def process_image(image):
    print("[OCR] Starting text detection")
    results = reader.readtext(image)
    if not results:
        print("[OCR] No text detected")
        speak("No text detected.")
        return

    texts = [text for (_, text, _) in results]
    joined = '. '.join(texts) + '.'
    print(f"[OCR] Detected combined text: {joined}")
    speak(joined)

def main():
    global capture_flag

    # Start key listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("[INIT] Key listener started")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Camera not available.")
        speak("Camera not available.")
        listener.stop()
        return
    print("[INIT] Camera opened")

    speak("Press C to capture and read text.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            speak("Failed to read frame.")
            break

        cv2.imshow("OCR - press C to capture", frame)

        if capture_flag:
            print("[FLOW] capture_flag detected!")
            capture_flag = False

            captured = frame.copy()
            cap.release()
            cv2.destroyAllWindows()
            print("[FLOW] Camera released and window closed")

            speak("Analyzing captured text.")
            process_image(captured)

            speak("Returning to voice interface.")
            time.sleep(0.5)

            listener.stop()
            print("[FLOW] Listener stopped, launching NLP script")

            # Relaunch the NLP script
            subprocess.run([sys.executable, "integratedvoicenlp.py"])
            return

        if cv2.waitKey(1) & 0xFF == 27:
            print("[INPUT] ESC pressed, exiting")
            break

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()
    print("[CLEANUP] Exited main loop and cleaned up")

if __name__ == "__main__":
    main()
