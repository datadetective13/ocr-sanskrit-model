import os
from pdf2image import convert_from_path
import cv2
import numpy as np

def enhance_image(image):
    """Enhance image quality for better segmentation"""
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Apply adaptive thresholding for better contrast
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def pdf_to_images(pdf_path, output_folder, dpi=300, skip_existing=True):
    """Convert PDF to enhanced images with high DPI"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_folder = os.path.join(output_folder, pdf_name)
    
    # Check if already processed
    if skip_existing and os.path.exists(pdf_output_folder):
        existing_images = [f for f in os.listdir(pdf_output_folder) if f.endswith('.png')]
        if existing_images:
            print(f"⏭️  Skipping {pdf_name} (already converted, {len(existing_images)} images found)")
            return len(existing_images)
    
    if not os.path.exists(pdf_output_folder):
        os.makedirs(pdf_output_folder)
    
    print(f"Converting {pdf_name} to images...")
    images = convert_from_path(pdf_path, dpi=dpi)
    
    for i, image in enumerate(images):
        # Enhance the image
        enhanced = enhance_image(image)
        
        # Save enhanced image
        image_path = os.path.join(pdf_output_folder, f'page_{i + 1:03d}.png')
        cv2.imwrite(image_path, enhanced)
        print(f'  Saved: page_{i + 1:03d}.png')
    
    return len(images)

def process_all_pdfs(pdf_directory, output_directory, skip_existing=True):
    """Process all PDFs in the manuscript directory"""
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory {pdf_directory} does not exist!")
        return
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    total_pages = 0
    skipped_pdfs = 0
    processed_pdfs = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        pages = pdf_to_images(pdf_path, output_directory, skip_existing=skip_existing)
        total_pages += pages
        
        # Check if it was skipped
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_output_folder = os.path.join(output_directory, pdf_name)
        if skip_existing and os.path.exists(pdf_output_folder):
            existing_images = [f for f in os.listdir(pdf_output_folder) if f.endswith('.png')]
            if len(existing_images) == pages:
                skipped_pdfs += 1
            else:
                processed_pdfs += 1
        else:
            processed_pdfs += 1
    
    print(f"\n{'='*60}")
    print(f"✓ PDF conversion complete!")
    print(f"✓ Total PDFs: {len(pdf_files)}")
    print(f"✓ Processed: {processed_pdfs}")
    print(f"⏭️  Skipped (already done): {skipped_pdfs}")
    print(f"✓ Total pages: {total_pages}")
    print(f"✓ Images saved to: {output_directory}")
    print(f"{'='*60}")

if __name__ == "__main__":
    pdf_directory = 'data/raw/manuscripts'
    output_directory = 'data/processed/images'
    
    process_all_pdfs(pdf_directory, output_directory)