import easyocr
import cv2
import pyttsx3
import keyboard
import time
import warnings
import sys  # ➡️ added for clean exit

# ✅ Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ Initialize TTS engine
engine = pyttsx3.init()

# ✅ Initialize EasyOCR reader with custom weights
reader = easyocr.Reader(
    ['en'],
    model_storage_directory='ocr_model',
    download_enabled=False
)

# ✅ Function to extract and read text
def process_image(image):
    results = reader.readtext(image)
    if not results:
        print("No text detected.")
        engine.say("No text detected.")
        engine.runAndWait()
        return

    for (_, text, conf) in results:
        print(f"Detected: {text} (Confidence: {conf:.2f})")

    # Prompt the user
    engine.say("Do you want me to read the text?")
    engine.runAndWait()
    print("[Y] Read | [N] Skip")

    while True:
        if keyboard.is_pressed('y'):
            engine.say("Reading the text now.")
            for (_, text, _) in results:
                engine.say(text)
            engine.runAndWait()
            break
        elif keyboard.is_pressed('n'):
            print("Skipped.")
            engine.say("Okay, skipping.")
            engine.runAndWait()
            break
        time.sleep(0.1)

# ✅ Main camera loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available.")
        return

    print("Press 'C' to capture and process text.")
    print("Hold 'C' for 3.5 seconds to return to NLP.")
    print("Press 'Q' to quit immediately.")

    hold_start = None  # ➡️ To track how long 'C' is held

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Display the live camera feed
        cv2.imshow("OCR - Press 'C' to capture", frame)

        # Handle key inputs
        if keyboard.is_pressed('c'):
            if hold_start is None:
                hold_start = time.time()
            else:
                held_time = time.time() - hold_start
                if held_time >= 3.5:
                    print("Returning to NLP...")
                    engine.say("Returning to voice control.")
                    engine.runAndWait()
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
        else:
            hold_start = None  # Reset if 'C' is released

        if keyboard.is_pressed('q'):
            print("Exiting...")
            engine.say("Shutting down OCR.")
            engine.runAndWait()
            break

        if keyboard.is_pressed('c') and hold_start is None:
            print("Captured frame for OCR.")
            captured_frame = frame.copy()
            process_image(captured_frame)
            time.sleep(1)  # Prevent multiple triggers

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
