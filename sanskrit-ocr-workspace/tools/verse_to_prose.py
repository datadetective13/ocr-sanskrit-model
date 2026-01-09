import os
import sys
import json
import re
from glob import glob
from transformers import AutoTokenizer, AutoModel
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

class KavyaGuruAligner:
    def __init__(self):
        print("Loading IndicBERT v2...")
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        self.model = AutoModel.from_pretrained("ai4bharat/indic-bert")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def is_verse(self, text):
        """Detect if text is verse (basic heuristic)"""
        # Check for verse markers or meter patterns
        verse_markers = ['॥', '।।']
        for marker in verse_markers:
            if marker in text:
                return True
        
        # Check line length consistency (verses have regular meter)
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            lengths = [len(line.strip()) for line in lines if line.strip()]
            if lengths and max(lengths) - min(lengths) < 5:
                return True
        
        return False
    
    def parse_shloka(self, text):
        """Parse verse into pada structure"""
        # Clean text
        text = text.replace('॥', '').replace('।', '')
        
        # Split into lines
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        
        # Typical shloka has 4 padas (quarters)
        pada_dict = {}
        for i, line in enumerate(lines[:4]):
            pada_dict[f'pada{i+1}'] = line
        
        return pada_dict
    
    def convert_to_anvaya(self, shloka_dict):
        """
        Convert verse to prose order (anvaya)
        This is a simplified version - full implementation requires
        Sanskrit grammatical analysis
        """
        # Concatenate all padas
        all_words = []
        for pada in ['pada1', 'pada2', 'pada3', 'pada4']:
            if pada in shloka_dict:
                all_words.extend(shloka_dict[pada].split())
        
        # Basic reordering (placeholder logic)
        # In reality, this needs dependency parsing and case analysis
        prose_order = ' '.join(all_words)
        
        return prose_order
    
    def embed_text(self, text):
        """Create IndicBERT embeddings"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    
    def process_file(self, input_file, output_file):
        """Process corrected OCR file and add prose alignment"""
        print(f"Processing: {os.path.basename(input_file)}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for page in data:
            for line in page['lines']:
                text = line.get('corrected_text', line.get('text', ''))
                
                if self.is_verse(text):
                    shloka = self.parse_shloka(text)
                    prose = self.convert_to_anvaya(shloka)
                    
                    line['is_verse'] = True
                    line['shloka_structure'] = shloka
                    line['prose_text'] = prose
                    
                    # Generate embedding for search
                    embedding = self.embed_text(prose)
                    line['embedding'] = embedding.tolist()
                else:
                    line['is_verse'] = False
                    line['prose_text'] = text
                    
                    # Generate embedding
                    embedding = self.embed_text(text)
                    line['embedding'] = embedding.tolist()
        
        # Save processed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved: {os.path.basename(output_file)}\n")

def main():
    print("="*60)
    print("VERSE-TO-PROSE ALIGNMENT (IndicBERT)")
    print("="*60)
    
    aligner = KavyaGuruAligner()
    
    input_dir = os.path.join(ROOT_DIR, 'data/processed/corrected_output')
    output_dir = os.path.join(ROOT_DIR, 'data/processed/aligned_output')
    os.makedirs(output_dir, exist_ok=True)
    
    corrected_files = glob(os.path.join(input_dir, '*_corrected.json'))
    
    print(f"\nFound {len(corrected_files)} files to align\n")
    
    for input_file in corrected_files:
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, basename.replace('_corrected.json', '_aligned.json'))
        
        aligner.process_file(input_file, output_file)
    
    print("="*60)
    print("✓ ALIGNMENT COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()