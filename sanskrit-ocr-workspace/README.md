# Sanskrit OCR Project

This project aims to develop a character recognition system for Sanskrit manuscripts using Optical Character Recognition (OCR) techniques. The workflow involves several phases, including data preparation, model training, and evaluation.

## Project Structure

```
sanskrit-ocr-workspace
├── data
│   ├── raw
│   │   └── manuscripts          # Original manuscript PDF files
│   ├── processed
│   │   ├── images              # Images extracted from PDFs
│   │   ├── lines               # Segmented line images
│   │   └── characters          # Segmented individual character images
│   └── labeled
│       ├── train               # Labeled training dataset
│       └── test                # Labeled testing dataset
├── tools
│   ├── annotation_tool.py      # GUI for manual annotation
│   ├── pdf_to_image.py         # Converts PDF files to images
│   ├── segmentation.py          # Segments images into lines and characters
│   └── augmentation.py          # Implements data augmentation techniques
├── models
│   ├── traditional_cv
│   │   └── feature_extraction.py # Traditional CV feature extraction
│   └── deep_learning
│       └── simple_cnn.py       # Simple CNN architecture for training
├── experiments
│   ├── baseline_ocr_test.py    # Tests existing OCR tools
│   └── results                  # Stores experiment results
├── notebooks
│   └── exploration.ipynb        # Jupyter notebook for exploratory analysis
├── requirements.txt             # Project dependencies
├── config.yaml                  # Configuration settings
└── README.md                    # Project documentation
```

## Workflow Overview

### Phase 1: Character Segmentation & Manual Labeling
1. **Extract and Segment Characters**: Convert PDF manuscripts to images, then segment these images into lines and individual characters.
2. **Create Labeled Dataset**: Manually label a minimum of 1000-1500 character images, ensuring coverage of all basic characters and conjuncts in the Sanskrit Devanagari script.

### Phase 2: Data Augmentation Strategy
- Utilize techniques such as rotation, scaling, Gaussian noise, and elastic transformations to generate additional training samples from the labeled dataset.

### Phase 3: Model Training Approaches
- **Option A**: Transfer Learning using pre-trained models like TrOCR-base or EasyOCR.
- **Option B**: Train a simple CNN from scratch, suitable for CPU training.

## Getting Started

1. **Clone the Repository**: 
   ```
   git clone <repository-url>
   cd sanskrit-ocr-workspace
   ```

2. **Install Dependencies**: 
   ```
   pip install -r requirements.txt
   ```

3. **Prepare Data**: Place your manuscript PDF files in the `data/raw/manuscripts` directory.

4. **Run Annotation Tool**: Use the annotation tool to label character images:
   ```
   python tools/annotation_tool.py
   ```

5. **Process Data**: Convert PDFs to images and segment them:
   ```
   python tools/pdf_to_image.py
   python tools/segmentation.py
   ```

6. **Train Model**: Choose your training approach and run the corresponding script in the `models` directory.

## Alternatives and Considerations
- Explore existing OCR tools like Tesseract or PaddleOCR for baseline performance.
- Consider a hybrid approach using traditional computer vision techniques for segmentation and feature extraction.

## Contribution
Feel free to contribute to this project by submitting issues or pull requests. Your feedback and improvements are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.