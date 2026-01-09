import os
import sys
import cv2
import numpy as np
from glob import glob
import json
import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

# Add parent directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.deep_learning.char_recognition import SanskritCharRecognizer

class OCRExtractor:
    def __init__(self, model_path, mappings_path):
        self.recognizer = SanskritCharRecognizer()
        self.recognizer.load_model(model_path, mappings_path)
    
    def extract_text_from_line(self, line_dir):
        """Extract text from all characters in a line directory"""
        char_files = sorted(glob(os.path.join(line_dir, 'char_*.png')))
        
        text = []
        confidences = []
        
        for char_file in char_files:
            char, confidence = self.recognizer.predict(char_file)
            text.append(char)
            confidences.append(confidence)
        
        return ''.join(text), np.mean(confidences)
    
    def extract_text_from_page(self, page_dir):
        """Extract text from all lines in a page"""
        # Find all line directories
        line_dirs = sorted([d for d in os.listdir(page_dir) 
                           if os.path.isdir(os.path.join(page_dir, d))])
        
        lines = []
        
        for line_dir in line_dirs:
            line_path = os.path.join(page_dir, line_dir)
            text, confidence = self.extract_text_from_line(line_path)
            
            if text:
                lines.append({
                    'line': line_dir,
                    'text': text,
                    'confidence': float(confidence)
                })
        
        return lines
    
    def extract_text_from_manuscript(self, manuscript_dir, output_file):
        """Extract text from entire manuscript"""
        page_dirs = sorted([d for d in os.listdir(manuscript_dir)
                           if os.path.isdir(os.path.join(manuscript_dir, d))])
        
        manuscript_text = []
        
        for page_dir in page_dirs:
            print(f"Processing {page_dir}...")
            page_path = os.path.join(manuscript_dir, page_dir)
            lines = self.extract_text_from_page(page_path)
            
            manuscript_text.append({
                'page': page_dir,
                'lines': lines
            })
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manuscript_text, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Extracted text saved to {output_file}")
        
        return manuscript_text

def preprocess_for_segmentation(image):
    """Enhanced preprocessing for better segmentation"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply additional denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Use Otsu's thresholding for better binarization
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove small noise using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary

def segment_lines(image, min_line_height=20, min_gap=8):
    """Improved line segmentation that handles varying spacing"""
    binary = preprocess_for_segmentation(image)
    
    # Horizontal projection
    horizontal_projection = np.sum(binary, axis=1)
    
    # Smooth the projection to handle noise
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(horizontal_projection, sigma=2)
    
    # Normalize
    if np.max(smoothed) > 0:
        smoothed = smoothed / np.max(smoothed)
    
    # Dynamic threshold - use percentile instead of mean
    threshold = np.percentile(smoothed[smoothed > 0], 20)
    
    # Find line boundaries
    line_regions = []
    in_line = False
    start = 0
    gap_counter = 0
    
    for i, val in enumerate(smoothed):
        if val > threshold:
            if not in_line:
                start = i
                in_line = True
            gap_counter = 0
        else:
            if in_line:
                gap_counter += 1
                if gap_counter >= min_gap:
                    if i - start >= min_line_height:
                        line_regions.append((start, i - gap_counter))
                    in_line = False
                    gap_counter = 0
    
    # Add last line
    if in_line and len(smoothed) - start >= min_line_height:
        line_regions.append((start, len(smoothed)))
    
    print(f"  📝 Detected {len(line_regions)} lines")
    
    # Extract line images with adaptive padding
    line_images = []
    for idx, (start, end) in enumerate(line_regions):
        line_height = end - start
        padding = max(5, int(line_height * 0.1))  # Adaptive padding
        
        start = max(0, start - padding)
        end = min(image.shape[0], end + padding)
        
        line_img = image[start:end, :]
        
        if line_img.size > 0 and line_height >= min_line_height:
            line_images.append(line_img)
    
    return line_images

def merge_broken_components(stats, labels, img_width, max_gap=15):
    """Merge components that are likely parts of the same character"""
    num_labels = len(stats)
    merged_components = []
    used = set()
    
    for i in range(1, num_labels):
        if i in used:
            continue
            
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        w1 = stats[i, cv2.CC_STAT_WIDTH]
        h1 = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Look for nearby components to merge
        merge_candidates = [i]
        
        for j in range(i + 1, num_labels):
            if j in used:
                continue
                
            x2 = stats[j, cv2.CC_STAT_LEFT]
            y2 = stats[j, cv2.CC_STAT_TOP]
            w2 = stats[j, cv2.CC_STAT_WIDTH]
            h2 = stats[j, cv2.CC_STAT_HEIGHT]
            
            # Check if components are horizontally close
            horizontal_gap = x2 - (x1 + w1)
            
            # Check if components are vertically aligned
            vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            
            # Merge if horizontally close and vertically overlapping
            if horizontal_gap >= 0 and horizontal_gap <= max_gap and vertical_overlap > 0:
                merge_candidates.append(j)
                used.add(j)
        
        if merge_candidates:
            # Calculate bounding box of merged component
            all_x = [stats[k, cv2.CC_STAT_LEFT] for k in merge_candidates]
            all_y = [stats[k, cv2.CC_STAT_TOP] for k in merge_candidates]
            all_right = [stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] for k in merge_candidates]
            all_bottom = [stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT] for k in merge_candidates]
            
            merged_x = min(all_x)
            merged_y = min(all_y)
            merged_w = max(all_right) - merged_x
            merged_h = max(all_bottom) - merged_y
            merged_area = sum([stats[k, cv2.CC_STAT_AREA] for k in merge_candidates])
            
            merged_components.append({
                'x': merged_x,
                'y': merged_y,
                'w': merged_w,
                'h': merged_h,
                'area': merged_area,
                'components': merge_candidates
            })
            used.update(merge_candidates)
    
    return merged_components

def segment_characters_advanced(line_image, min_width=5, min_height=12):
    """Advanced character segmentation with component merging"""
    binary = preprocess_for_segmentation(line_image)
    
    # Apply morphological closing to connect broken strokes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    # Merge broken components
    merged = merge_broken_components(stats, labels, line_image.shape[1])
    
    # Also keep unmerged components
    used_in_merge = set()
    for comp in merged:
        used_in_merge.update(comp['components'])
    
    character_boxes = []
    
    # Add merged components
    for comp in merged:
        if comp['w'] >= min_width and comp['h'] >= min_height:
            character_boxes.append((comp['x'], comp['y'], comp['w'], comp['h'], comp['area']))
    
    # Add unmerged components
    for i in range(1, num_labels):
        if i not in used_in_merge:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if w >= min_width and h >= min_height and area > 25:
                # Filter unrealistic aspect ratios
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 <= aspect_ratio <= 4.0:
                    character_boxes.append((x, y, w, h, area))
    
    # Sort by x-coordinate
    character_boxes.sort(key=lambda box: box[0])
    
    # Extract character images
    character_images = []
    for x, y, w, h, area in character_boxes:
        # Adaptive padding based on character size
        padding = max(4, int(min(w, h) * 0.15))
        
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(line_image.shape[1], x + w + padding)
        y_end = min(line_image.shape[0], y + h + padding)
        
        char_img = line_image[y_start:y_end, x_start:x_end]
        
        # Normalize character image
        char_img = normalize_character_image(char_img)
        if char_img is not None:
            character_images.append(char_img)
    
    return character_images

def normalize_character_image(char_img, target_size=(128, 128)):
    """Normalize with better quality for varied writing"""
    if char_img is None or char_img.size == 0:
        return None
    
    h, w = char_img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    
    # Convert to grayscale if needed
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive binarization for varying ink density
    binary = cv2.adaptiveThreshold(char_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Invert if needed (text should be black on white)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    
    # Calculate scaling
    scale = min(target_size[0] * 0.8 / h, target_size[1] * 0.8 / w)
    
    if scale <= 0 or scale > 5:
        return None
    
    new_h, new_w = int(h * scale), int(w * scale)
    
    if new_h < 1 or new_w < 1:
        return None
    
    try:
        # Use appropriate interpolation
        if scale > 1:
            resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas
        canvas = np.ones(target_size, dtype=np.uint8) * 255
        
        # Center the character
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Apply slight morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        
        return canvas
    except Exception as e:
        print(f"⚠️ Normalization failed: {e}")
        return None

def process_image_to_characters(image_path, output_base_dir, skip_existing=True):
    """Process with improved segmentation and skip logic"""
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    parent_folder = os.path.basename(os.path.dirname(image_path))
    
    # Create output directory path
    chars_dir = os.path.join(output_base_dir, 'characters', parent_folder, image_name)
    
    # ✅ CHECK IF ALREADY PROCESSED
    if skip_existing and os.path.exists(chars_dir):
        existing_chars = [f for f in os.listdir(chars_dir) if f.startswith('char_') and f.endswith('.png')]
        if len(existing_chars) > 0:
            print(f"  ⏭️  Skipping {image_name} (already segmented, {len(existing_chars)} chars found)")
            return len(existing_chars)
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"  ❌ Failed to load: {image_name}")
        return 0
    
    print(f"  📄 {image_name}")
    
    # Segment lines
    line_images = segment_lines(image)
    
    if len(line_images) == 0:
        print(f"    ❌ No lines found")
        return 0
    
    # Create output directories
    lines_dir = os.path.join(output_base_dir, 'lines', parent_folder)
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(chars_dir, exist_ok=True)
    
    total_chars = 0
    
    for line_idx, line_img in enumerate(line_images):
        # Save line
        line_path = os.path.join(lines_dir, f'{image_name}_line_{line_idx:03d}.png')
        cv2.imwrite(line_path, line_img)
        
        # Segment characters
        char_images = segment_characters_advanced(line_img)
        
        # Save characters
        for char_idx, char_img in enumerate(char_images):
            if char_img is not None:
                char_path = os.path.join(chars_dir, f'char_L{line_idx:03d}_C{char_idx:03d}.png')
                cv2.imwrite(char_path, char_img)
                total_chars += 1
        
        print(f"    Line {line_idx + 1}: {len(char_images)} chars")
    
    return total_chars

def process_all_images(images_dir, output_base_dir='data/processed', skip_existing=True):
    """Process all images with resume capability"""
    
    print("\n" + "="*70)
    print("CHARACTER SEGMENTATION")
    print("="*70)
    
    if skip_existing:
        print("⏭️  Skip mode: ON (already segmented images will be skipped)")
    else:
        print("♻️  Skip mode: OFF (all images will be re-segmented)")
    
    image_folders = [d for d in os.listdir(images_dir) 
                     if os.path.isdir(os.path.join(images_dir, d))]
    
    if not image_folders:
        print(f"❌ No folders in {images_dir}")
        return
    
    total_characters = 0
    processed_images = 0
    skipped_images = 0
    failed_images = 0
    
    for folder in image_folders:
        folder_path = os.path.join(images_dir, folder)
        image_files = sorted(glob(os.path.join(folder_path, '*.png')))
        
        print(f"\n📁 {folder} ({len(image_files)} images)")
        print("-" * 70)
        
        for img_path in image_files:
            try:
                chars_count = process_image_to_characters(img_path, output_base_dir, skip_existing)
                
                if chars_count > 0:
                    # Check if it was actually processed or skipped
                    if skip_existing:
                        # Check if "Skipping" message was printed
                        image_name = os.path.splitext(os.path.basename(img_path))[0]
                        parent_folder = os.path.basename(os.path.dirname(img_path))
                        chars_dir = os.path.join(output_base_dir, 'characters', parent_folder, image_name)
                        
                        # If directory already existed with files, it was skipped
                        if os.path.exists(chars_dir):
                            existing_before = len([f for f in os.listdir(chars_dir) if f.startswith('char_')])
                            if existing_before == chars_count:
                                skipped_images += 1
                            else:
                                processed_images += 1
                        else:
                            processed_images += 1
                    else:
                        processed_images += 1
                    
                    total_characters += chars_count
                    print(f"  ✅ Total: {chars_count} characters\n")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}\n")
                failed_images += 1
    
    print(f"\n{'='*70}")
    print(f"✅ SEGMENTATION COMPLETE!")
    print(f"  • Newly processed: {processed_images} images")
    print(f"  • Skipped (already done): {skipped_images} images")
    if failed_images > 0:
        print(f"  • Failed: {failed_images} images")
    print(f"  • Total characters: {total_characters}")
    print(f"{'='*70}")

class ByT5Corrector:
    def __init__(self, model_name="google/byt5-small"):
        """
        Initialize ByT5 for Sanskrit OCR post-correction
        You can fine-tune this model on Sanskrit correction pairs
        """
        print(f"Loading {model_name}...")
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def correct_text(self, text, max_length=512):
        """Correct OCR errors in text"""
        # Prepare input
        input_text = f"correct: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", 
                               max_length=max_length, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate correction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text
    
    def correct_ocr_file(self, input_file, output_file):
        """Correct OCR errors in a JSON file"""
        print(f"Loading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Correcting text...")
        for page in data:
            for line in page['lines']:
                original_text = line['text']
                corrected_text = self.correct_text(original_text)
                line['corrected_text'] = corrected_text
                print(f"  Original:  {original_text}")
                print(f"  Corrected: {corrected_text}\n")
        
        # Save corrected text
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Corrected text saved to {output_file}")

if __name__ == "__main__":
    model_path = 'models/trained/char_recognizer.keras'
    mappings_path = 'models/trained/char_mappings.json'
    
    extractor = OCRExtractor(model_path, mappings_path)
    
    # Process all manuscripts
    chars_dir = 'data/processed/characters'
    output_dir = 'data/processed/ocr_output'
    os.makedirs(output_dir, exist_ok=True)
    
    manuscripts = [d for d in os.listdir(chars_dir)
                  if os.path.isdir(os.path.join(chars_dir, d))]
    
    for manuscript in manuscripts:
        manuscript_path = os.path.join(chars_dir, manuscript)
        output_file = os.path.join(output_dir, f"{manuscript}_ocr.json")
        
        print(f"\n{'='*60}")
        print(f"Processing manuscript: {manuscript}")
        print(f"{'='*60}")
        
        extractor.extract_text_from_manuscript(manuscript_path, output_file)
    
    corrector = ByT5Corrector()
    
    input_dir = 'data/processed/ocr_output'
    output_dir = 'data/processed/corrected_output'
    os.makedirs(output_dir, exist_ok=True)
    
    ocr_files = glob(os.path.join(input_dir, '*_ocr.json'))
    
    for ocr_file in ocr_files:
        basename = os.path.basename(ocr_file)
        output_file = os.path.join(output_dir, basename.replace('_ocr.json', '_corrected.json'))
        
        print(f"\n{'='*60}")
        print(f"Processing: {basename}")
        print(f"{'='*60}\n")
        
        corrector.correct_ocr_file(ocr_file, output_file)