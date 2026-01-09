import os
import sys
from glob import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from tools.segmentation import ByT5Corrector

def main():
    print("="*60)
    print("ByT5 OCR POST-CORRECTION")
    print("="*60)
    
    corrector = ByT5Corrector(model_name="google/byt5-small")
    
    input_dir = os.path.join(ROOT_DIR, 'data/processed/ocr_output')
    output_dir = os.path.join(ROOT_DIR, 'data/processed/corrected_output')
    os.makedirs(output_dir, exist_ok=True)
    
    ocr_files = glob(os.path.join(input_dir, '*_ocr.json'))
    
    print(f"\nFound {len(ocr_files)} OCR files to correct\n")
    
    for ocr_file in ocr_files:
        basename = os.path.basename(ocr_file)
        output_file = os.path.join(output_dir, basename.replace('_ocr.json', '_corrected.json'))
        
        print(f"Processing: {basename}")
        corrector.correct_ocr_file(ocr_file, output_file)
        print(f"✓ Saved to: {os.path.basename(output_file)}\n")
    
    print("="*60)
    print("✓ ByT5 CORRECTION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()