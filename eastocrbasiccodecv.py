import cv2
import easyocr
import pyttsx3
import sys  # Allows switching between modes

# ✅ Function to Detect and Open First Available Camera
def get_camera_index():
    for index in range(5):  # Check first 5 indexes (0,1,2,3,4)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return None  # No available camera

# ✅ Try to Auto-Detect Camera
camera_index = get_camera_index()
if camera_index is None:
    print("No available camera found. Exiting...")
    exit()

# ✅ Initialize Camera
cap = cv2.VideoCapture(camera_index)

# ✅ Initialize OCR & TTS
reader = easyocr.Reader(['en'], model_storage_directory='ocr_model')
engine = pyttsx3.init()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed not available. Exiting...")
        break

    cv2.imshow("Press 'C' to Capture | 'O' to Switch to Object Detection | 'Q' to Quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        print("Image Captured! Processing text...")
        cv2.waitKey(1)  # Ensure frame updates

        # ✅ Process the current frame
        results = reader.readtext(frame)
        extracted_text = " ".join([text for (_, text, _) in results])

        if extracted_text.strip():  # ✅ Only if text is detected
            print("Detected Text:")
            print(extracted_text)

            # ✅ Ask via TTS if the user wants the text read aloud
            engine.say("I have found a text. Press Y for Yes or N for No.")
            engine.runAndWait()

            while True:  # ✅ Wait for user input (Y/N)
                key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
                if key == ord('y'):
                    print("Reading text...")
                    engine.say(extracted_text)
                    engine.runAndWait()
                    break
                elif key == ord('n'):
                    print("Skipping text reading.")
                    break

        else:
            print("No text detected.")
            engine.say("No text detected. Try again.")
            engine.runAndWait()

        print("Ready for another capture. Press 'C' to capture again or 'Q' to quit.")

    elif key == ord('o'):  # ✅ Switch to YOLO Object Detection
        print("Switching to Object Detection...")
        cv2.destroyAllWindows()
        sys.exit(1)  # Exit with status 1 to signal `main.py` to switch to YOLO

    elif key == ord('q'):
        print("Exiting...")
        break
 
cap.release()
cv2.destroyAllWindows()
# import cv2
# import easyocr
# import pyttsx3
# import sys  # Allows switching between modes
# import numpy as np

# # ✅ Function to Detect and Open First Available Camera
# def get_camera_index():
#     for index in range(5):  # Check first 5 indexes (0,1,2,3,4)
#         cap = cv2.VideoCapture(index)
#         if cap.isOpened():
#             cap.release()
#             return index
#     return None  # No available camera

# # ✅ Try to Auto-Detect Camera
# camera_index = get_camera_index()
# if camera_index is None:
#     print("No available camera found. Exiting...")
#     exit()

# # ✅ Initialize Camera
# cap = cv2.VideoCapture(camera_index)

# # ✅ Initialize OCR & TTS
# reader = easyocr.Reader(['en'], model_storage_directory='ocr_model')
# engine = pyttsx3.init()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Camera feed not available. Exiting...")
#         break

#     cv2.imshow("Press 'C' to Capture | 'O' to Switch to Object Detection | 'Q' to Quit", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('c'):
#         print("Image Captured! Processing text...")
#         cv2.waitKey(1)  # Ensure frame updates

#         # --- Begin Preprocessing for Better OCR Performance ---
#         # Convert captured frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Apply histogram equalization to improve contrast
#         equ = cv2.equalizeHist(gray)
#         # Apply Gaussian blur to reduce noise
#         blur = cv2.GaussianBlur(equ, (5, 5), 1)
#         # Apply manual thresholding (adjust threshold value as needed)
#         _, processed = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
#         # Optionally, you can display the processed image for debugging:
#         # cv2.imshow("Processed", processed)
#         # --- End Preprocessing ---

#         # Use the preprocessed image for OCR
#         results = reader.readtext(processed)
#         extracted_text = " ".join([text for (_, text, _) in results])

#         if extracted_text.strip():  # ✅ Only if text is detected
#             print("Detected Text:")
#             print(extracted_text)

#             # ✅ Ask via TTS if the user wants the text read aloud
#             engine.say("I have found a text. Press Y for Yes or N for No.")
#             engine.runAndWait()

#             while True:  # ✅ Wait for user input (Y/N)
#                 key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
#                 if key == ord('y'):
#                     print("Reading text...")
#                     engine.say(extracted_text)
#                     engine.runAndWait()
#                     break
#                 elif key == ord('n'):
#                     print("Skipping text reading.")
#                     break
#         else:
#             print("No text detected.")
#             engine.say("No text detected. Try again.")
#             engine.runAndWait()

#         print("Ready for another capture. Press 'C' to capture again or 'Q' to quit.")

#     elif key == ord('o'):  # ✅ Switch to YOLO Object Detection
#         print("Switching to Object Detection...")
#         cv2.destroyAllWindows()
#         sys.exit(1)  # Exit with status 1 to signal `main.py` to switch to YOLO

#     elif key == ord('q'):
#         print("Exiting...")
#         break
 
# cap.release()
# cv2.destroyAllWindows()
