# DocLayout-YOLO Layout Detection Pipeline

A simplified, standalone implementation of document layout detection using DocLayout-YOLO. This pipeline can process both images and PDF files to detect various layout elements in documents.

## Features

- **DocLayout-YOLO Integration**: Uses the state-of-the-art DocLayout-YOLO model for accurate layout detection
- **Flexible Input**: Supports single images, PDF files, or directories containing multiple files
- **Multiple Formats**: Handles PNG, JPG, JPEG images and PDF documents
- **Visualization**: Generates annotated images showing detected layout elements
- **Easy Configuration**: YAML-based configuration for easy customization
- **Fallback Support**: Falls back to ultralytics YOLO if DocLayout-YOLO is not available

## Detected Layout Elements

The model can detect the following layout elements:

- **Title**: Document titles and headings
- **Plain Text**: Regular text content
- **Figure**: Images, charts, diagrams
- **Figure Caption**: Captions for figures
- **Table**: Data tables
- **Table Caption**: Captions for tables
- **Table Footnote**: Footnotes related to tables
- **Isolate Formula**: Mathematical formulas
- **Formula Caption**: Captions for formulas
- **Abandon**: Elements to be ignored

## Installation

1. **Clone or download this pipeline**:
   ```bash
   # If part of a larger repository
   cd layout_detection_pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install DocLayout-YOLO** (recommended):
   ```bash
   pip install doclayout-yolo
   ```

   Or use ultralytics as fallback:
   ```bash
   pip install ultralytics
   ```

4. **Download the model**:
   Download the DocLayout-YOLO model from:
   [https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/doclayout_yolo_ft.pt](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/doclayout_yolo_ft.pt)

   Place it in a `models/` directory or update the path in `config.yaml`.

## Usage

### Quick Start

1. **Update configuration**:
   Edit `config.yaml` to set your model path and input/output directories.

2. **Run detection**:
   ```bash
   python run_detection.py
   ```

### Command Line Options

```bash
# Use custom input and output
python run_detection.py --input /path/to/input --output /path/to/output

# Use different model
python run_detection.py --model /path/to/model.pt

# Use GPU
python run_detection.py --device cuda

# Use custom config file
python run_detection.py --config my_config.yaml

# Specify JSON output format
python run_detection.py --format simple
python run_detection.py --format coco

# Process specific pages from PDF
python run_detection.py --input document.pdf --pages "1,3,5"
python run_detection.py --input document.pdf --pages "1-5"
python run_detection.py --input document.pdf --pages "1,3-7,10"

# Get PDF information
python run_detection.py --input document.pdf --info
```

### Programmatic Usage

```python
from layout_detector import LayoutDetector

# Initialize detector
detector = LayoutDetector(
    model_path="models/doclayout_yolo_ft.pt",
    device='cpu',  # or 'cuda' for GPU
    conf_thres=0.25,
    iou_thres=0.45,
    img_size=1024,
    visualize=True
)

# Process input (image, PDF, or directory)
results = detector.predict("input_path", "output_directory")

# Process specific pages from PDF
results = detector.predict("document.pdf", "output_directory", page_numbers=[1, 3, 5])

# Get PDF information
pdf_info = detector.get_pdf_info("document.pdf")
print(f"PDF has {pdf_info['total_pages']} pages")

# Save results in different formats
detector.save_results_json(results, "output_directory", "detailed")
detector.save_results_json(results, "output_directory", "simple")
detector.save_results_json(results, "output_directory", "coco")

# Access results
for result in results:
    print(f"Image: {result['image_id']}")
    print(f"Detected elements: {len(result['detections'])}")
    for detection in result['detections']:
        bbox = detection['bbox']
        print(f"  - {detection['class_name']}: {detection['confidence']:.3f}")
        print(f"    Box: [{bbox['x1']:.0f}, {bbox['y1']:.0f}, {bbox['x2']:.0f}, {bbox['y2']:.0f}]")
```

## Input Formats

The pipeline supports various input formats:

- **Single Image**: `image.jpg`, `document.png`
- **Single PDF**: `document.pdf`
- **Directory**: Folder containing images and/or PDFs

## Page Selection for PDFs

The pipeline supports processing specific pages from PDF files, which is useful for:
- **Targeted Analysis**: Process only pages of interest
- **Performance**: Faster processing and reduced storage
- **Testing**: Quick validation on specific pages

### Page Selection Syntax

```bash
# Single page
--pages "1"

# Multiple specific pages
--pages "1,3,5"

# Page range
--pages "1-5"          # Pages 1, 2, 3, 4, 5

# Mixed selection
--pages "1,3-7,10"     # Pages 1, 3, 4, 5, 6, 7, 10
```

### PDF Information

Get PDF information before processing:

```bash
python run_detection.py --input document.pdf --info
```

Output:
```
PDF Information:
  File: document.pdf
  Total pages: 15
  Available pages: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
```

### Page Selection Examples

```bash
# Process first page only
python run_detection.py --input document.pdf --pages "1"

# Process first 5 pages
python run_detection.py --input document.pdf --pages "1-5"

# Process specific pages with custom output
python run_detection.py --input document.pdf --pages "1,5,10" --output results_selected

# Process page range with GPU
python run_detection.py --input document.pdf --pages "10-15" --device cuda
```

### Programmatic Page Selection

```python
# Get PDF information
pdf_info = detector.get_pdf_info("document.pdf")
print(f"Total pages: {pdf_info['total_pages']}")

# Process specific pages
results = detector.predict("document.pdf", "output", page_numbers=[1, 3, 5])

# Process page range
page_range = list(range(1, 6))  # Pages 1-5
results = detector.predict("document.pdf", "output", page_numbers=page_range)

# Process last page
last_page = pdf_info['total_pages']
results = detector.predict("document.pdf", "output", page_numbers=[last_page])
```

## Output

The pipeline generates:

1. **Visualization Images**: Annotated images showing detected layout elements (if enabled)
2. **JSON Results**: Multiple JSON formats with detailed detection results
3. **Console Output**: Summary of detections

### Output Structure

```
output_directory/
├── detection_results_detailed.json    # Comprehensive results with metadata
├── detection_results_simple.json      # Clean, minimal format
├── detection_results_coco.json        # COCO-style format (optional)
├── image1_layout.png                  # Visualized results (if enabled)
├── image2_layout.png
└── ...
```

### JSON Output Formats

#### 1. Detailed Format (default)
Comprehensive format with all available information:

```json
{
  "metadata": {
    "model_info": {
      "model_path": "models/doclayout_yolo_ft.pt",
      "device": "cpu",
      "confidence_threshold": 0.25,
      "iou_threshold": 0.45,
      "image_size": 1024
    },
    "class_mapping": {"0": "title", "1": "plain text", ...},
    "total_images": 1,
    "total_detections": 5
  },
  "results": [
    {
      "image_id": "document_page_0001",
      "image_info": {
        "source": "document.pdf",
        "total_detections": 5
      },
      "detections": [
        {
          "bbox": {
            "x1": 100.0, "y1": 50.0, "x2": 300.0, "y2": 150.0,
            "width": 200.0, "height": 100.0, "area": 20000.0
          },
          "class_id": 0,
          "class_name": "title",
          "confidence": 0.95
        }
      ]
    }
  ]
}
```

#### 2. Simple Format
Clean, minimal format for easy integration:

```json
[
  {
    "image_id": "document_page_0001",
    "detections": [
      {
        "bbox": [100.0, 50.0, 300.0, 150.0],
        "class": "title",
        "confidence": 0.95
      }
    ]
  }
]
```

#### 3. COCO Format
Standard computer vision format:

```json
{
  "images": [
    {"id": 1, "file_name": "document_page_0001", "width": 0, "height": 0}
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 0,
      "bbox": [100.0, 50.0, 200.0, 100.0],
      "area": 20000.0, "iscrowd": 0, "score": 0.95
    }
  ],
  "categories": [
    {"id": 0, "name": "title", "supercategory": "layout_element"}
  ]
}
```

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
model:
  model_path: "models/doclayout_yolo_ft.pt"
  device: "cpu"  # or "cuda"
  img_size: 1024
  conf_thres: 0.25
  iou_thres: 0.45

paths:
  input: "input"
  output: "output"

visualization:
  enabled: true
  alpha: 0.3
```

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- PIL/Pillow
- PyMuPDF (for PDF processing)
- DocLayout-YOLO or Ultralytics YOLO
- NumPy

## Performance Tips

1. **Use GPU**: Set `device: "cuda"` for faster processing
2. **Batch Processing**: Process multiple files in a directory
3. **Adjust Thresholds**: Tune `conf_thres` and `iou_thres` for your use case
4. **Image Size**: Larger `img_size` may improve accuracy but slower processing

## Troubleshooting

1. **Model not found**: Ensure the model path in config.yaml is correct
2. **CUDA errors**: Use `device: "cpu"` if GPU issues occur
3. **Memory issues**: Reduce `img_size` or process files individually
4. **Import errors**: Install missing dependencies from requirements.txt

## License

This pipeline is based on the PDF-Extract-Kit project. Please refer to the original project's license for usage terms.
