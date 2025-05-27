import cv2
import torch
import pyttsx3
import warnings
import time
import subprocess
import sys

# === Suppress Warnings ===
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# === Initialize TTS engine ===
engine = pyttsx3.init()

def speak(text):
    print(f"[SPEAK]: {text}")
    engine.say(text)
    engine.runAndWait()

# === Set device and load model ===
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
FOCAL_LENGTH = 700  # Adjust based on your camera
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

speak("Live detection started. Press and hold the Button for 3 seconds to return.")

last_announce_time = 0
c_key_pressed_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        speak("Failed to read frame.")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    margin = int(frame_width * 0.1)  # 10% margin for "middle"

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

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} - {int(distance)}cm ({side})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Speak if object is close enough and delay passed
        if not announced and distance <= THRESHOLD_DISTANCE_CM and (current_time - last_announce_time) > ANNOUNCE_DELAY:
            speak(f"{label} detected at {int(distance)} centimeters {side}")
            last_announce_time = current_time
            announced = True

    # === Show frame ===
    cv2.imshow("Live Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Hold 'C' to return after 3 seconds
    if key == ord('c'):
        if c_key_pressed_time is None:
            c_key_pressed_time = time.time()
        elif time.time() - c_key_pressed_time >= 3:
            speak("Going back to listening mode.")
            break
    else:
        c_key_pressed_time = None

# === Cleanup ===
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()

# === Relaunch NLP ===
subprocess.run([sys.executable, "integratedvoicenlp.py"])
