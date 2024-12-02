import cv2
import os
from mtcnn import MTCNN
from tqdm import tqdm  # For progress bar
from multiprocessing import Pool, cpu_count
import sys

# Global detector variable
detector = None


def init_detector():
    """
    Initializer function for each worker process.
    Initializes the MTCNN detector globally within the process.
    """
    global detector
    detector = MTCNN()


def process_image(args):
    """
    Processes a single image: detects faces and writes labels.

    Args:
        args (tuple): A tuple containing:
            - image_file (str): The image file name.
            - input_dir (str): Directory containing input images.
            - output_dir (str): Directory to save annotation files.
            - conf_threshold (float): Confidence threshold for detections.
    """
    image_file, input_dir, output_dir, conf_threshold = args
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read {image_path}. Skipping.", file=sys.stderr)
        return

    # Convert BGR to RGB as MTCNN expects RGB images
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    detections = detector.detect_faces(rgb_image)
    faces = []
    for detection in detections:
        confidence = detection['confidence']
        if confidence > conf_threshold:
            x, y, width, height = detection['box']
            startX = max(0, x)
            startY = max(0, y)
            endX = startX + width
            endY = startY + height
            # Calculate YOLO format coordinates
            bbox_width = width / image.shape[1]
            bbox_height = height / image.shape[0]
            center_x = (startX + endX) / 2 / image.shape[1]
            center_y = (startY + endY) / 2 / image.shape[0]
            faces.append([0, center_x, center_y, bbox_width, bbox_height])  # [class_id, cx, cy, w, h]

    if faces:
        # Create a corresponding .txt file
        txt_filename = os.path.splitext(image_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            for face in faces:
                line = ' '.join([f"{val:.6f}" if isinstance(val, float) else str(val) for val in face])
                f.write(line + '\n')
    return


def annotate_images_mtcnn_multithreaded(input_dir, output_dir, conf_threshold=0.5, num_workers=None):
    """
    Annotates images using MTCNN in a multiprocessing environment.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where annotation files will be saved.
        conf_threshold (float): Confidence threshold to filter detections.
        num_workers (int, optional): Number of worker processes. Defaults to number of CPU cores.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Create argument tuples for each image
    args = [(image_file, input_dir, output_dir, conf_threshold) for image_file in image_files]

    # Determine number of worker processes
    if num_workers is None:
        num_workers = cpu_count()

    # Initialize the multiprocessing pool
    with Pool(processes=num_workers, initializer=init_detector) as pool:
        # Use imap_unordered for better performance and progress tracking
        for _ in tqdm(pool.imap_unordered(process_image, args), total=len(args), desc="Annotating Images with MTCNN"):
            pass

    print(f"Annotation completed. Annotations saved in {output_dir}")


if __name__ == "__main__":
    # Paths (Replace with your actual paths)
    input_images_dir = "data/images/val"  # e.g., "./images/my_photos"
    annotations_output_dir = "data/labels/val"  # e.g., "./labels/labels_mtcnn"
    confidence_threshold = 0.3  # Adjust based on your needs

    # Optionally, specify the number of worker processes
    num_workers = 16  # Example: 4 parallel processes

    annotate_images_mtcnn_multithreaded(input_images_dir, annotations_output_dir, confidence_threshold)