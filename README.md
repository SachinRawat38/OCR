# Enhanced OCR for Receipt Processing

A powerful and adaptive OCR (Optical Character Recognition) solution designed specifically for processing receipts and similar documents, with multiple preprocessing techniques to maximize text extraction accuracy.

## Features

- **Multiple Preprocessing Techniques**: Automatically applies and combines results from various image preprocessing methods to achieve optimal text extraction
- **Adaptive Processing**: Analyzes image characteristics to determine the best preprocessing approach
- **Deskewing**: Automatically corrects tilted images to improve text recognition
- **Duplicate Removal**: Intelligently filters duplicate text detections 
- **Confidence-Based Selection**: Keeps only the highest confidence detection for each text element
- **Structured Data Extraction**: Attempts to identify key receipt elements such as store names, dates, totals, and items
- **Visual Debugging**: Displays all preprocessing steps and annotated results for better understanding

## Requirements

- Python 3.6+
- OpenCV
- EasyOCR
- NumPy
- Matplotlib
- scikit-image

## Installation

```bash
pip install easyocr opencv-python numpy matplotlib scikit-image
```

## Usage

### Basic Usage

```python
from enhanced_ocr import process_receipt

# Process a receipt image
results, structured_data, annotated_image = process_receipt('path/to/receipt.jpg')

# Display results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 12))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

### Functions Overview

- `initialize_ocr(languages=['en'])`: Initializes the EasyOCR reader
- `read_image(image_path)`: Reads an image from a file path or array
- `auto_adjust_gamma(image, gamma=1.0)`: Automatically adjusts gamma correction
- `deskew_image(image)`: Straightens tilted text
- `adaptive_preprocessing(image)`: Applies multiple preprocessing techniques
- `detect_text_with_multiple_preprocessings(reader, image)`: Combines OCR results from different preprocessing methods
- `draw_text_boxes(image, results)`: Annotates the image with bounding boxes and text
- `display_preprocessing_steps(original_image, preprocessed_images, annotated_image)`: Shows visual steps
- `extract_generic_text_data(results, image_shape)`: Extracts structured data from OCR results
- `format_extracted_data(structured_data)`: Formats extracted data for human-readable output
- `process_receipt(image_path, show_preprocessing=True)`: Main function to process a receipt image

## Preprocessing Methods

The system applies multiple preprocessing techniques to maximize text extraction accuracy:

1. **Grayscale Conversion**
2. **Binary Thresholding** (Otsu's method)
3. **Gaussian Blur + Binary Threshold**
4. **Adaptive Thresholding**
5. **Local Thresholding**
6. **Gamma Correction**
7. **Noise Reduction**
8. **Contrast Enhancement** (CLAHE)
9. **Edge Detection + Dilation**
10. **Deskewing**

## Output

The script produces:

- **OCR Results**: Raw text detection with bounding boxes and confidence scores
- **Structured Data**: An attempt to identify key receipt elements
- **Annotated Image**: Visual representation of detected text
- **Processing Summary**: Formatted text output of all detected information

## Example

```python
from enhanced_ocr import process_receipt

# Process the receipt
results, structured_data, annotated_image = process_receipt('images/receipt.jpg')
```

## Structured Data Format

The extracted structured data includes:

- `text_blocks`: All text blocks in reading order
- `possible_headers`: Text at the top (likely headers/store info)
- `possible_totals`: Lines with numbers that might be totals
- `possible_items`: Lines that might be items
- `possible_dates`: Text that looks like dates
- `possible_barcodes`: Text that looks like numeric codes
- `location_map`: Map dividing receipt into regions
- `possible_store_name`: Best guess at the store name
