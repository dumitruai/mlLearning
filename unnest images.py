import os
import shutil
import random


def split_dataset(images_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, files in splits.items():
        split_image_dir = os.path.join(output_dir, 'data', split)
        split_annotation_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(split_image_dir, exist_ok=True)
        os.makedirs(split_annotation_dir, exist_ok=True)
        for file in files:
            shutil.copy(os.path.join(images_dir, file), os.path.join(split_image_dir, file))

    print("Dataset split completed.")


if __name__ == "__main__":
    source_images_dir = "data/train"
    destination_dir = "data"
    split_dataset(source_images_dir, destination_dir)