import onnxruntime as ort
import numpy as np
import cv2
import pyttsx3
import keyboard
import time
import warnings
import yaml

# ✅ Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ Load class names from data.yaml
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names'] if 'names' in data else []

# ✅ Load class names from your custom dataset
class_names = load_class_names('yaml_files/datasetfinal_5.yaml')

# ✅ Load ONNX model (CPU only)
session = ort.InferenceSession('yolomodels/yolov5n.onnx', providers=['CPUExecutionProvider'])

# ✅ Initialize TTS
engine = pyttsx3.init()

# ✅ Preprocess frame
def preprocess(frame, img_size=640):
    img = cv2.resize(frame, (img_size, img_size))
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_input /= 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    return img_input

# ✅ Main loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No available camera found.")
        return

    last_announced_object = None
    y_pressed_time = None
    y_hold_duration = 3  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2

        input_tensor = preprocess(frame)
        outputs = session.run(None, {'images': input_tensor})
        detections = outputs[0][0]  # [N, ...] where first 6 values are [x1, y1, x2, y2, conf, cls_id]

        closest_object = None
        min_distance = float('inf')

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]  # ✅ Only take the first 6 values
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            distance = np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2)

            obj_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"class_{int(cls_id)}"

            if distance < min_distance:
                min_distance = distance
                closest_object = (obj_name, x1, y1, x2, y2)

        # ✅ Announce object if 'Y' is held
        if keyboard.is_pressed('y'):
            if y_pressed_time is None:
                y_pressed_time = time.time()
            elif (time.time() - y_pressed_time) >= y_hold_duration:
                if closest_object:
                    obj_name = closest_object[0]
                    if obj_name != last_announced_object:
                        print(f"Announcing: {obj_name}")
                        engine.say(obj_name)
                    else:
                        print(f"Still detecting: {obj_name}")
                        engine.say(f"Still detecting {obj_name}")
                    engine.runAndWait()
                    last_announced_object = obj_name
                y_pressed_time = None
        else:
            y_pressed_time = None

        # ✅ Draw detections
        if closest_object:
            obj_name, x1, y1, x2, y2 = closest_object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ Crosshair at center
        cv2.drawMarker(frame, (center_x, center_y), (255, 0, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        cv2.imshow("ONNX YOLOv5 Detection", frame)

        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            print("Exiting...")
            break

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
