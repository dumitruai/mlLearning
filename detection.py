import sys
import os
import cv2
import torch
import numpy as np

# Add YOLOv5 directory to Python path
sys.path.insert(0, os.path.abspath('yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device


def load_model(weights_path, device='cpu'):
    """
    Loads the YOLOv5 model from the local repository.

    Args:
        weights_path (str): Path to the trained model weights (.pt file).
        device (str): 'cpu' or 'cuda' for GPU acceleration.

    Returns:
        model: Loaded YOLOv5 model.
    """
    # Initialize the device
    device = select_device(device)

    # Load the model
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names


def real_time_face_detection(model, stride, names, device='cpu', conf_threshold=0.5):
    """
    Performs real-time face detection using the webcam.

    Args:
        model: Loaded YOLOv5 model.
        stride: Model stride.
        names: Class names.
        device (str): 'cpu' or 'cuda' for GPU acceleration.
        conf_threshold (float): Confidence threshold for detections.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Convert frame to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)

        # Resize and pad image while meeting stride-multiple constraints
        img_size = 640
        img_resized = cv2.resize(img, (img_size, img_size))
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, 0)
        img_input = torch.from_numpy(img_input).to(device)
        img_input = img_input.float()

        # Inference
        with torch.no_grad():
            preds = model(img_input, augment=False, visualize=False)

        # NMS
        preds = non_max_suppression(preds, conf_threshold, 0.45, classes=None, agnostic=False)

        # Process detections
        for det in preds:
            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_boxes(img_input.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    x1, y1, x2, y2 = map(int, xyxy)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put confidence score
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-Time Face Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Path to your trained YOLOv5 model
    weights_path = 'face_detector_yolov5s3/weights/best.pt'

    # Choose device: 'cuda' for GPU, 'cpu' otherwise
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model, stride, names = load_model(weights_path, device)

    # Start real-time detection
    real_time_face_detection(model, stride, names, device, conf_threshold=0.5)