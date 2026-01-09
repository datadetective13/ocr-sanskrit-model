import os
import sys
import shutil
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from tools.pdf_to_image import process_all_pdfs
from tools.enhanced_preprocessing import HistoricalManuscriptEnhancer
from tools.segmentation import process_all_images

def clean_processed_data(root_dir):
    """Remove all processed data for a fresh start"""
    print("\n" + "="*70)
    print("🗑️  CLEANING OLD PROCESSED DATA")
    print("="*70)
    
    dirs_to_clean = [
        os.path.join(root_dir, 'data', 'processed', 'images'),
        os.path.join(root_dir, 'data', 'processed', 'images_enhanced'),
        os.path.join(root_dir, 'data', 'processed', 'lines'),
        os.path.join(root_dir, 'data', 'processed', 'characters'),
        os.path.join(root_dir, 'data', 'processed', 'ocr_output'),
        os.path.join(root_dir, 'data', 'processed', 'corrected_output'),
        os.path.join(root_dir, 'data', 'processed', 'aligned_output'),
    ]
    
    removed_count = 0
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                file_count = sum([len(files) for _, _, files in os.walk(dir_path)])
                shutil.rmtree(dir_path)
                print(f"  ✓ Removed: {os.path.relpath(dir_path, root_dir)} ({file_count} files)")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {dir_path}: {e}")
        else:
            print(f"  ⏭️  Skipped: {os.path.relpath(dir_path, root_dir)} (doesn't exist)")
    
    print(f"\n✅ Cleaned {removed_count} directories")
    print("="*70)

def ensure_directory(path):
    """Ensure a path exists as a directory"""
    if os.path.exists(path):
        if os.path.isfile(path):
            print(f"⚠ Warning: '{path}' exists as a file. Removing...")
            os.remove(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def run_complete_pipeline(clean=False):
    """Run the complete pipeline with resume capability
    
    Args:
        clean (bool): If True, remove all old processed data before starting
    """
    
    print("="*70)
    print("SANSKRIT OCR - ENHANCED PROCESSING PIPELINE")
    print("For Historical Manuscripts with Shirorekha")
    print("="*70)
    print(f"Working directory: {ROOT_DIR}")
    
    if clean:
        print("\n⚠️  CLEAN MODE: Will delete all processed data")
    else:
        print("\n✅ RESUME MODE: Will skip already processed files")
    
    # Clean old data if requested
    if clean:
        response = input("\n⚠️  This will DELETE all processed data. Continue? (yes/no): ")
        if response.lower() == 'yes':
            clean_processed_data(ROOT_DIR)
        else:
            print("❌ Cleaning cancelled. Exiting...")
            return
    
    # Define paths
    pdf_dir = os.path.join(ROOT_DIR, 'data', 'raw', 'manuscripts')
    images_dir = os.path.join(ROOT_DIR, 'data', 'processed', 'images')
    enhanced_dir = os.path.join(ROOT_DIR, 'data', 'processed', 'images_enhanced')
    
    # Ensure directories exist
    print("\nChecking and creating directories...")
    ensure_directory(os.path.join(ROOT_DIR, 'data'))
    ensure_directory(os.path.join(ROOT_DIR, 'data', 'raw'))
    ensure_directory(pdf_dir)
    ensure_directory(os.path.join(ROOT_DIR, 'data', 'processed'))
    ensure_directory(images_dir)
    
    # Step 1: PDF to Images
    print("\n" + "="*70)
    print("[STEP 1/4] Converting PDFs to images (300 DPI)...")
    print("="*70)
    
    # Use skip_existing unless clean mode
    skip_existing = not clean
    process_all_pdfs(pdf_dir, images_dir, skip_existing=skip_existing)
    
    # Check if images were created
    if not os.path.exists(images_dir) or not os.path.isdir(images_dir) or not os.listdir(images_dir):
        print("\n❌ No images found to process.")
        print(f"Please add PDF files to: {pdf_dir}")
        return
    
    # Step 2: Enhanced Image Processing
    print("\n" + "="*70)
    print("[STEP 2/4] Enhancing images for old manuscripts...")
    print("  • Deblurring")
    print("  • Denoising")
    print("  • Background normalization")
    print("  • Contrast enhancement")
    print("  • Shirorekha-aware binarization")
    print("="*70)
    
    # Use skip logic in enhancement
    enhancer = HistoricalManuscriptEnhancer(skip_existing=skip_existing)
    enhancer.process_directory(images_dir, enhanced_dir)
    
    # Step 3: Segment lines and characters
    print("\n" + "="*70)
    print("[STEP 3/4] Segmenting lines and characters...")
    print("="*70)
    output_base_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    
    # Use ENHANCED images for segmentation with skip logic
    process_all_images(enhanced_dir, output_base_dir, skip_existing=skip_existing)
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    
    # Count results
    characters_dir = os.path.join(ROOT_DIR, 'data', 'processed', 'characters')
    if os.path.exists(characters_dir):
        total_chars = sum([len([f for f in os.listdir(os.path.join(root, d)) 
                               if f.endswith('.png')])
                          for root, dirs, files in os.walk(characters_dir)
                          for d in dirs])
        print(f"\n📊 Statistics:")
        print(f"  • Total character images: {total_chars}")
        print(f"  • Location: {characters_dir}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Review character images in: {characters_dir}")
    print(f"2. Start labeling using: python tools/label_characters.py")
    print(f"3. Label at least 1,000 images across 30+ character classes")
    print(f"4. Train model: python models/deep_learning/char_recognition.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sanskrit OCR Processing Pipeline')
    parser.add_argument('--clean', action='store_true', 
                       help='Remove all old processed data before running (fresh start)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-processing of all files (no skip)')
    args = parser.parse_args()
    
    # If --force is specified, it's equivalent to --clean
    clean = args.clean or args.force
    
    run_complete_pipeline(clean=clean)