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

# Access results
for result in results:
    print(f"Image: {result['image_id']}")
    print(f"Detected elements: {len(result['boxes'])}")
    for i, class_name in enumerate(result['class_names']):
        print(f"  - {class_name}: {result['scores'][i]:.3f}")
```

## Input Formats

The pipeline supports various input formats:

- **Single Image**: `image.jpg`, `document.png`
- **Single PDF**: `document.pdf`
- **Directory**: Folder containing images and/or PDFs

## Output

The pipeline generates:

1. **Visualization Images**: Annotated images showing detected layout elements (if enabled)
2. **JSON Results**: Detailed detection results including bounding boxes, classes, and confidence scores
3. **Console Output**: Summary of detections

### Output Structure

```
output_directory/
├── detection_results.json          # Detailed results in JSON format
├── image1_layout.png              # Visualized results (if enabled)
├── image2_layout.png
└── ...
```

### JSON Results Format

```json
[
  {
    "image_id": "document_page_0001",
    "boxes": [[x1, y1, x2, y2], ...],
    "classes": [0, 1, 3, ...],
    "scores": [0.95, 0.87, 0.92, ...],
    "class_names": ["title", "plain text", "figure", ...]
  }
]
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
