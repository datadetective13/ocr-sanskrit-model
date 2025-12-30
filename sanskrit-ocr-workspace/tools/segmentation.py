import os
import cv2
import numpy as np

def segment_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((1, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        line_image = image[y:y+h, x:x+w]
        line_images.append(line_image)

    return line_images

def segment_characters(line_image):
    gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((1, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        character_image = line_image[y:y+h, x:x+w]
        character_images.append(character_image)

    return character_images

def process_image(image_path):
    image = cv2.imread(image_path)
    line_images = segment_lines(image)

    all_character_images = []
    for line_image in line_images:
        character_images = segment_characters(line_image)
        all_character_images.extend(character_images)

    return all_character_images

if __name__ == "__main__":
    input_image_path = "path/to/your/image.jpg"  # Update this path
    character_images = process_image(input_image_path)

    output_dir = "data/processed/characters"
    os.makedirs(output_dir, exist_ok=True)

    for i, char_img in enumerate(character_images):
        cv2.imwrite(os.path.join(output_dir, f"char_{i}.png"), char_img)