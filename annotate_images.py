import cv2
import os
from tqdm import tqdm  # For progress bar


def detect_faces_opencv_dnn(net, image, conf_threshold=0.5):
    (h, w) = image.shape[:2]
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            # Ensure the bounding boxes fall within the image dimensions
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)
            # Calculate YOLO format coordinates
            bbox_width = (endX - startX) / w
            bbox_height = (endY - startY) / h
            center_x = (startX + endX) / 2 / w
            center_y = (startY + endY) / 2 / h
            faces.append({
                "bbox": [center_x, center_y, bbox_width, bbox_height],
                "confidence": float(confidence)
            })
    return faces


def annotate_images_opencv_dnn(input_dir, output_dir, model_path, config_path, conf_threshold=0.5):
    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Annotating Images"):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_path}. Skipping.")
            continue
        faces = detect_faces_opencv_dnn(net, image, conf_threshold)
        if faces:
            # Create a corresponding .txt file
            txt_filename = os.path.splitext(image_file)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, 'w') as f:
                for face in faces:
                    class_id = 0  # Assuming 'face' is class 0
                    bbox = face['bbox']
                    # Ensure values are between 0 and 1
                    bbox = [min(max(coord, 0), 1) for coord in bbox]
                    line = f"{class_id} {' '.join([f'{coord:.6f}' for coord in bbox])}\n"
                    f.write(line)
    print(f"Annotation completed. Annotations saved in {output_dir}")


if __name__ == "__main__":
    # Paths (Replace with your actual paths)
    input_images_dir = "data/images/train"  # e.g., "./images/my_photos"
    annotations_output_dir = "data/labels/train"  # e.g., "./labels/labels"
    model_file = "models/Res10_300x300_SSD_iter_140000.caffemodel"
    config_file = "models/Resnet_SSD_deploy.prototxt"
    confidence_threshold = 0.5  # Adjust based on your needs

    annotate_images_opencv_dnn(input_images_dir, annotations_output_dir, model_file, config_file, confidence_threshold)