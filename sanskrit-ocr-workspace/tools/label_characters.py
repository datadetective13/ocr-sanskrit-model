from flask import Flask, render_template, request, jsonify
import os
import shutil
import json
from glob import glob
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARS_DIR = os.path.join(ROOT_DIR, 'data', 'processed', 'characters')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'labeled', 'train')
PROGRESS_FILE = os.path.join(ROOT_DIR, 'data', 'labeled', 'labeling_progress.json')

# Character classes
CHARACTER_CLASSES = [
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः',
    'क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह',
    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः',
    'क्ष', 'त्र', 'ज्ञ', 'श्र',
    '।', '॥', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९'
]

# Global state
current_images = []
current_index = 0
progress = {'labeled': [], 'skipped': [], 'labels': {}}

def load_progress():
    global progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            progress = json.load(f)

def save_progress():
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_images():
    global current_images
    all_images = []
    for root, dirs, files in os.walk(CHARS_DIR):
        for file in files:
            if file.startswith('char_') and file.endswith('.png'):
                all_images.append(os.path.join(root, file))
    
    labeled_set = set(progress.get('labeled', []))
    current_images = [img for img in all_images if img not in labeled_set]
    print(f"Loaded {len(current_images)} unlabeled images")

def image_to_base64(img_path):
    """Convert image to base64 for web display"""
    try:
        img = Image.open(img_path)
        # Convert grayscale to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for better display (optional)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

@app.route('/')
def index():
    return render_template('labeler.html', 
                         characters=CHARACTER_CLASSES,
                         total=len(current_images),
                         labeled=len(progress.get('labeled', [])),
                         skipped=len(progress.get('skipped', [])))

@app.route('/get_image')
def get_image():
    global current_index
    
    if current_index >= len(current_images):
        return jsonify({'done': True})
    
    img_path = current_images[current_index]
    img_base64 = image_to_base64(img_path)
    
    if img_base64 is None:
        return jsonify({'error': 'Failed to load image'})
    
    return jsonify({
        'done': False,
        'image': img_base64,
        'filename': os.path.basename(img_path),
        'full_path': img_path,
        'index': current_index + 1,
        'total': len(current_images),
        'labeled': len(progress.get('labeled', [])),
        'skipped': len(progress.get('skipped', []))
    })

@app.route('/save_label', methods=['POST'])
def save_label():
    global current_index
    
    data = request.json
    char = data.get('character')
    
    if not char or current_index >= len(current_images):
        return jsonify({'success': False, 'error': 'Invalid request'})
    
    img_path = current_images[current_index]
    
    # Create target directory
    target_dir = os.path.join(OUTPUT_DIR, char)
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate filename
    existing_files = len([f for f in os.listdir(target_dir) if f.endswith('.png')])
    target_filename = f"{char}_{existing_files:05d}.png"
    target_path = os.path.join(target_dir, target_filename)
    
    # Copy image
    shutil.copy2(img_path, target_path)
    
    # Update progress
    progress['labeled'].append(img_path)
    progress.setdefault('labels', {})[img_path] = char
    save_progress()
    
    print(f"✓ Labeled: {os.path.basename(img_path)} as '{char}'")
    
    current_index += 1
    
    return jsonify({'success': True, 'next': current_index < len(current_images)})

@app.route('/skip', methods=['POST'])
def skip():
    global current_index
    
    if current_index >= len(current_images):
        return jsonify({'success': False})
    
    img_path = current_images[current_index]
    progress['skipped'].append(img_path)
    save_progress()
    
    print(f"⏭ Skipped: {os.path.basename(img_path)}")
    
    current_index += 1
    
    return jsonify({'success': True, 'next': current_index < len(current_images)})

@app.route('/previous', methods=['POST'])
def previous():
    global current_index
    
    if current_index > 0:
        current_index -= 1
        return jsonify({'success': True})
    
    return jsonify({'success': False})

@app.route('/statistics')
def statistics():
    class_counts = {}
    for char_class in CHARACTER_CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, char_class)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            if count > 0:
                class_counts[char_class] = count
    
    return jsonify({
        'total_labeled': len(progress.get('labeled', [])),
        'total_skipped': len(progress.get('skipped', [])),
        'unique_classes': len(class_counts),
        'class_counts': sorted(class_counts.items(), key=lambda x: -x[1])
    })

if __name__ == '__main__':
    # Initialize
    load_progress()
    load_images()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("   🕉️  SANSKRIT CHARACTER LABELING TOOL - WEB VERSION")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   • Total character images: {len(current_images) + len(progress.get('labeled', []))}")
    print(f"   • Already labeled: {len(progress.get('labeled', []))}")
    print(f"   • Already skipped: {len(progress.get('skipped', []))}")
    print(f"   • Remaining to label: {len(current_images)}")
    print(f"\n🌐 Starting web server...")
    print(f"   → Open in browser: http://localhost:5000")
    print(f"   → Or from network: http://{os.popen('hostname').read().strip()}.local:5000")
    print(f"\n⌨️  Keyboard shortcuts:")
    print(f"   • Enter = Save & Next")
    print(f"   • S = Skip")
    print(f"   • A / ← = Previous")
    print(f"   • → = Next (after saving)")
    print(f"\n🛑 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)