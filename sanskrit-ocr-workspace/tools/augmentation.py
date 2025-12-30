import os
import cv2
import numpy as np
from glob import glob

def augment_image(image):
    # Rotation
    angle = np.random.uniform(-5, 5)
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))

    # Scaling
    scale = np.random.uniform(0.9, 1.1)
    scaled = cv2.resize(rotated, None, fx=scale, fy=scale)

    # Gaussian noise
    noise = np.random.normal(0, 25, scaled.shape).astype(np.uint8)
    noisy = cv2.add(scaled, noise)

    # Blur variations
    blurred = cv2.GaussianBlur(noisy, (5, 5), 0)

    # Brightness/contrast changes
    brightness = np.random.randint(-30, 30)
    contrast = np.random.uniform(0.8, 1.2)
    adjusted = cv2.convertScaleAbs(blurred, alpha=contrast, beta=brightness)

    return adjusted

def augment_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob(os.path.join(input_dir, '*.png'))  # Assuming images are in PNG format
    for image_path in image_paths:
        image = cv2.imread(image_path)
        for i in range(10):  # Generate 10 augmented images per original image
            augmented_image = augment_image(image)
            base_name = os.path.basename(image_path).split('.')[0]
            output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
            cv2.imwrite(output_path, augmented_image)

if __name__ == "__main__":
    input_characters_dir = '../data/labeled/train'  # Adjust path as necessary
    output_augmented_dir = '../data/labeled/train/augmented'  # Adjust path as necessary
    augment_dataset(input_characters_dir, output_augmented_dir)