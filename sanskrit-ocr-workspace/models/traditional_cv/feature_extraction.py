import cv2
import numpy as np

def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)
    return features.flatten()

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def extract_features_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Resize image for consistent feature extraction
    image = cv2.resize(image, (128, 128))
    
    # Extract HOG features
    hog_features = extract_hog_features(image)
    
    # Extract SIFT features
    sift_features = extract_sift_features(image)
    
    return {
        'hog_features': hog_features,
        'sift_features': sift_features
    }

def process_character_images(character_image_paths):
    features = {}
    for path in character_image_paths:
        features[path] = extract_features_from_image(path)
    return features