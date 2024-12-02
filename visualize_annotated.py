import cv2
import os
from tqdm import tqdm


def visualize_annotations(image_dir, annotations_dir, num_samples=20):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sampled_images = image_files[:num_samples]

    for image_file in tqdm(sampled_images, desc="Visualizing Annotations"):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_path}. Skipping.")
            continue
        txt_filename = os.path.splitext(image_file)[0] + '.txt'
        txt_path = os.path.join(annotations_dir, txt_filename)
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id, center_x, center_y, width, height = map(float, parts)
            img_h, img_w, _ = image.shape
            # Convert YOLO format to bounding box coordinates
            x_center = center_x * img_w
            y_center = center_y * img_h
            bbox_width = width * img_w
            bbox_height = height * img_h
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)
            # Draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Put class label
            cv2.putText(image, f"Face: {class_id}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the image
        cv2.imshow('Annotated Image', image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_dir = "data/images/test"  # e.g., "./images/my_photos"
    annotations_dir = "data/labels/test"  # e.g., "./labels/labels"
    visualize_annotations(images_dir, annotations_dir, num_samples=20)