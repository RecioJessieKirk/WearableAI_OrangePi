import onnxruntime as ort
import numpy as np
import cv2

# Use CPU execution
providers = ['CPUExecutionProvider']
session = ort.InferenceSession('yolomodels/yolov5n.onnx', providers=providers)

# Load and preprocess the image
def preprocess(image_path, img_size=640):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_input /= 255.0  # normalize to 0-1
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
    return img, img_input

# Run inference
def infer(image_path):
    orig_img, input_tensor = preprocess(image_path)
    outputs = session.run(None, {'images': input_tensor})

    # YOLOv5 ONNX output: [num_detections, 6] -> [x1, y1, x2, y2, conf, class]
    preds = outputs[0][0]

    for det in preds:
        x1, y1, x2, y2, conf, cls_id = det
        if conf > 0.4:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_img, f'Class {int(cls_id)}: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("Detection", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
infer('test_images/sample.jpg')
