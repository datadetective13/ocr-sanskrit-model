import cv2
import numpy as np
from scipy import ndimage
from skimage import restoration, filters, morphology
from skimage.restoration import denoise_tv_chambolle
import os
from glob import glob

class HistoricalManuscriptEnhancer:
    """
    Enhanced preprocessing for old Sanskrit manuscripts with:
    - Blur removal (deconvolution)
    - Noise reduction
    - Contrast enhancement
    - Shirorekha preservation
    - Background normalization
    """
    
    def __init__(self, skip_existing=True):
        self.debug = False
        self.skip_existing = skip_existing
    
    def enhance_manuscript_image(self, image_path, output_path=None):
        """Complete enhancement pipeline"""
        
        # ✅ CHECK IF ALREADY PROCESSED
        if self.skip_existing and output_path and os.path.exists(output_path):
            # Verify file is not empty/corrupted
            if os.path.getsize(output_path) > 100:
                print(f"⏭️  Skipping: {os.path.basename(image_path)} (already enhanced)")
                return cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"Enhancing: {os.path.basename(image_path)}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load: {image_path}")
            return None
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Step 1: Deblur (Richardson-Lucy deconvolution)
        print("  1/7 Deblurring...")
        deblurred = self.deblur_image(gray)
        
        # Step 2: Denoise
        print("  2/7 Denoising...")
        denoised = self.advanced_denoise(deblurred)
        
        # Step 3: Background normalization
        print("  3/7 Normalizing background...")
        normalized = self.normalize_background(denoised)
        
        # Step 4: Contrast enhancement (CLAHE)
        print("  4/7 Enhancing contrast...")
        enhanced = self.enhance_contrast(normalized)
        
        # Step 5: Shirorekha-aware binarization
        print("  5/7 Binarizing (Shirorekha-aware)...")
        binary = self.shirorekha_aware_binarization(enhanced)
        
        # Step 6: Morphological cleanup
        print("  6/7 Cleaning up...")
        cleaned = self.morphological_cleanup(binary)
        
        # Step 7: Shirorekha enhancement
        print("  7/7 Enhancing Shirorekha...")
        final = self.enhance_shirorekha(cleaned)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, final)
            print(f"  ✅ Saved to: {output_path}")
        
        return final
    
    def deblur_image(self, image, iterations=10):
        """Richardson-Lucy deconvolution for blur removal"""
        psf = self._create_gaussian_psf(size=5, sigma=1.5)
        
        try:
            deblurred = restoration.richardson_lucy(
                image.astype(float) / 255.0,
                psf,
                num_iter=iterations
            )
            deblurred = (deblurred * 255).astype(np.uint8)
        except Exception as e:
            print(f"    ⚠️  Deblur failed, using original: {e}")
            deblurred = image
        
        return deblurred
    
    def _create_gaussian_psf(self, size=5, sigma=1.0):
        """Create Gaussian Point Spread Function"""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return psf / np.sum(psf)
    
    def advanced_denoise(self, image):
        """Multi-stage denoising"""
        img_float = image.astype(float) / 255.0
        denoised_tv = denoise_tv_chambolle(img_float, weight=0.1)
        denoised = (denoised_tv * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            denoised, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
        return denoised
    
    def normalize_background(self, image):
        """Remove uneven illumination"""
        kernel_size = max(image.shape) // 15
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        normalized = cv2.subtract(image, background)
        normalized = cv2.add(normalized, 30)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        return normalized
    
    def enhance_contrast(self, image):
        """CLAHE contrast enhancement"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def shirorekha_aware_binarization(self, image):
        """Adaptive binarization preserving Shirorekha"""
        window_size = 51
        k = 0.2
        binary = self._sauvola_threshold(image, window_size, k)
        
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def _sauvola_threshold(self, image, window_size, k=0.2, R=128):
        """Sauvola's binarization method"""
        mean = cv2.boxFilter(image.astype(float), -1, (window_size, window_size))
        sqr_mean = cv2.boxFilter(image.astype(float)**2, -1, (window_size, window_size))
        std = np.sqrt(sqr_mean - mean**2)
        threshold = mean * (1 + k * ((std / R) - 1))
        binary = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary
    
    def morphological_cleanup(self, binary):
        """Remove small noise"""
        kernel_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        kernel_close = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        
        output = np.zeros_like(cleaned)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 15:
                output[labels == i] = 255
        
        return output
    
    def enhance_shirorekha(self, binary):
        """Enhance Shirorekha (horizontal header line)"""
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        dilate_kernel = np.ones((3, 3), np.uint8)
        enhanced_lines = cv2.dilate(detected_lines, dilate_kernel, iterations=1)
        result = cv2.bitwise_or(binary, enhanced_lines)
        return result
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory with skip logic"""
        print("\n" + "="*70)
        print("ENHANCED IMAGE PROCESSING FOR HISTORICAL MANUSCRIPTS")
        print("="*70)
        
        if self.skip_existing:
            print("⏭️  Skip mode: ON (existing enhanced images will be skipped)")
        else:
            print("♻️  Skip mode: OFF (all images will be re-processed)")
        
        folders = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
        
        total_processed = 0
        total_skipped = 0
        total_failed = 0
        
        for folder in folders:
            input_folder_path = os.path.join(input_dir, folder)
            output_folder_path = os.path.join(output_dir, folder)
            
            image_files = sorted(glob(os.path.join(input_folder_path, '*.png')))
            
            print(f"\n📁 {folder} ({len(image_files)} images)")
            print("-" * 70)
            
            folder_processed = 0
            folder_skipped = 0
            
            for img_path in image_files:
                output_path = os.path.join(
                    output_folder_path,
                    'enhanced_' + os.path.basename(img_path)
                )
                
                try:
                    result = self.enhance_manuscript_image(img_path, output_path)
                    if result is not None:
                        # Check if it was skipped or processed
                        if self.skip_existing and os.path.exists(output_path):
                            # File existed before this call
                            if "Skipping" in str(result):
                                folder_skipped += 1
                                total_skipped += 1
                            else:
                                folder_processed += 1
                                total_processed += 1
                        else:
                            folder_processed += 1
                            total_processed += 1
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    total_failed += 1
            
            if folder_processed > 0:
                print(f"  ✅ Processed {folder_processed} new images")
            if folder_skipped > 0:
                print(f"  ⏭️  Skipped {folder_skipped} existing images")
        
        print("\n" + "="*70)
        print(f"✅ COMPLETE!")
        print(f"  • Newly enhanced: {total_processed} images")
        print(f"  • Skipped (already done): {total_skipped} images")
        if total_failed > 0:
            print(f"  • Failed: {total_failed} images")
        print(f"📁 Output directory: {output_dir}")
        print("="*70)

def main():
    """Run enhanced preprocessing with resume capability"""
    enhancer = HistoricalManuscriptEnhancer(skip_existing=True)
    
    input_dir = 'data/processed/images'
    output_dir = 'data/processed/images_enhanced'
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("Please run pipeline first: python tools/pipeline.py")
        return
    
    enhancer.process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()