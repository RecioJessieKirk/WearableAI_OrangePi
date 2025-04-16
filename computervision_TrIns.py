# from ultralytics import YOLO
# import cv2
# import pyttsx3
# import time
# import keyboard  # For non-blocking key detection
# import numpy as np
# import sys  # Allows exiting the script
# import torch 
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

# # ✅ Initialize YOLO model
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda'
# print(f"Using device: {device}")

# # Load model
# model = torch.hub.load(
#     r'Data Cleaning Training/yolov5', 
#     'custom', 
#     path=r'Data Cleaning Training/trainedyolomodel/best.pt',
#     source='local',
#     force_reload=True
# ).to(device)



# # ✅ Initialize Text-to-Speech (TTS)
# engine = pyttsx3.init()

# # ✅ Track object announcements
# last_announced_object = None
# y_pressed_time = None  # Track when 'Y' is first pressed
# y_hold_duration = 3  # Seconds to trigger announcement

# try:
#     # ✅ Start webcam and process detections
#     for result in model.predict(source=camera_index, show=True, stream=True):
#         frame = result.orig_img  # Get the original video frame
#         h, w, _ = frame.shape  # Get frame dimensions
#         center_x, center_y = w // 2, h // 2  # Calculate the center of the frame

#         detected_objects = []
#         closest_object = None
#         min_distance = float('inf')  # Initialize with a large value

#         # ✅ Process detected objects
#         if result.boxes:
#             for box in result.boxes:
#                 obj_name = model.names[int(box.cls)]  # Get object name
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = (y1 + y2) // 2

#                 # Compute distance to frame center  
#                 distance = np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2)

#                 # Update closest object
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_object = obj_name

#         # ✅ Detect if 'Y' is being held down
#         if keyboard.is_pressed('y'):
#             if y_pressed_time is None:
#                 y_pressed_time = time.time()  # Start timing when 'Y' is first pressed

#             # Check if 'Y' is held for the required duration
#             elif (time.time() - y_pressed_time) >= y_hold_duration:
#                 if closest_object:
#                     if closest_object != last_announced_object:  # New object detected
#                         print(f"Announcing object: {closest_object}")
#                         engine.say(f"{closest_object}")
#                     else:  # Same object detected
#                         print(f"Still detecting: {closest_object}")
#                         engine.say(f"Still detecting {closest_object}")

#                     engine.runAndWait()
#                     last_announced_object = closest_object  # Update last announced object

#                 y_pressed_time = None  # Reset timer after speaking
#         else:
#             y_pressed_time = None  # Reset if 'Y' is released

#         # ✅ Detect if 'O' is pressed (Switch to OCR)
#         if keyboard.is_pressed('o'):
#             print("Switching to OCR mode...")
#             cv2.destroyAllWindows()
#             sys.exit(1)  # Exit the script to signal `main.py` to switch to OCR

#         # ✅ Draw bounding boxes and display the frame
#         if result.boxes:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
#                 obj_name = model.names[int(box.cls)]
#                 color = (0, 255, 0) if obj_name == closest_object else (0, 0, 255)  # Highlight closest object in green

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, obj_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # ✅ Draw a crosshair at the center of the screen
#         cv2.drawMarker(frame, (center_x, center_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

#         # ✅ Display the frame
#         cv2.imshow("YOLO Detection", frame)

#         # ✅ Ensure OpenCV window stays open
#         if cv2.getWindowProperty("YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:
#             print("Window closed. Stopping detection.")
#             break

#         # ✅ Allow exiting the loop with 'ESC'
#         if keyboard.is_pressed('esc'):
#             print("User pressed 'ESC'. Stopping detection.")
#             break

#         # ✅ Allow quitting with 'Q'
#         if keyboard.is_pressed('q'):
#             print("User pressed 'Q'. Stopping detection.")
#             break

#         cv2.waitKey(1)  # Ensures OpenCV window updates properly

# except KeyboardInterrupt:
#     print("Program interrupted manually.")

# finally:
#     cv2.destroyAllWindows()
#     print("Detection stopped, and resources cleaned up.")

import torch
import cv2
import numpy as np
import time
from PIL import Image

# Function to load YOLOv5 model
def load_model():
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load custom YOLOv5 model from local path
    model = torch.hub.load(
        r'Data Cleaning Training/yolov5', 
        'custom', 
        path=r'Data Cleaning Training/yolomodels/yolov5n.onnx',
        source='local',
        force_reload=True
    ).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

# Function to perform inference on a frame
def detect_objects(model, frame):
    # Convert frame to RGB (YOLOv5 expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(frame_rgb)
    
    # Perform inference
    results = model(img)
    
    return results

# Function to draw bounding boxes on frame
def draw_boxes(frame, results):
    # Get detections
    detections = results.pandas().xyxy[0]
    
    # Draw bounding boxes
    for i, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        cls = detection['class']
        name = detection['name']
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Load YOLOv5 model
    print("Loading custom YOLOv5 model...")
    model = load_model()
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Starting detection. Press 'q' to quit.")
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Perform detection
        results = detect_objects(model, frame)
        
        # Draw bounding boxes
        frame = draw_boxes(frame, results)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('YOLOv5 Detection', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()