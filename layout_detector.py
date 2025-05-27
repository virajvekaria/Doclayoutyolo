"""
Simplified DocLayout-YOLO Layout Detection Pipeline

This module provides a standalone implementation for document layout detection
using DocLayout-YOLO model. It can process both images and PDF files.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from typing import List, Union, Dict, Any, Optional


class LayoutDetector:
    """
    A simplified layout detection class using DocLayout-YOLO.
    """

    def __init__(self, model_path: str, device: str = 'cpu', conf_thres: float = 0.25,
                 iou_thres: float = 0.45, img_size: int = 1024, visualize: bool = True):
        """
        Initialize the LayoutDetector.

        Args:
            model_path (str): Path to the DocLayout-YOLO model file (.pt)
            device (str): Device to run inference on ('cpu' or 'cuda')
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            img_size (int): Input image size for the model
            visualize (bool): Whether to save visualization results
        """
        self.model_path = model_path
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.visualize = visualize

        # Class mapping for layout elements
        self.id_to_names = {
            0: 'title',
            1: 'plain text',
            2: 'abandon',
            3: 'figure',
            4: 'figure_caption',
            5: 'table',
            6: 'table_caption',
            7: 'table_footnote',
            8: 'isolate_formula',
            9: 'formula_caption'
        }

        # Load the model
        self._load_model()

        # Initialize NMS function if needed
        if self.iou_thres > 0:
            import torchvision
            self.nms_func = torchvision.ops.nms

    def _load_model(self):
        """Load the DocLayout-YOLO model."""
        try:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(self.model_path)
            print("Loaded DocLayout-YOLO model successfully")
        except (ImportError, AttributeError) as e:
            print(f"DocLayout-YOLO not available, falling back to ultralytics YOLO: {e}")
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print("Loaded ultralytics YOLO model successfully")
            except ImportError:
                raise ImportError("Neither doclayout_yolo nor ultralytics is available. Please install one of them.")

    def _load_pdf_images(self, pdf_path: str, dpi: int = 144,
                        page_numbers: Optional[List[int]] = None) -> List[Image.Image]:
        """
        Load images from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file
            dpi (int): DPI for rendering PDF pages
            page_numbers (Optional[List[int]]): Specific page numbers to process (1-indexed).
                                              If None, processes all pages.

        Returns:
            List[Image.Image]: List of PIL Images from PDF pages
        """
        images = []
        doc = fitz.open(pdf_path)

        # Determine which pages to process
        if page_numbers is None:
            # Process all pages
            pages_to_process = range(len(doc))
        else:
            # Process specific pages (convert from 1-indexed to 0-indexed)
            pages_to_process = [p - 1 for p in page_numbers if 1 <= p <= len(doc)]
            if not pages_to_process:
                print(f"Warning: No valid page numbers found in PDF with {len(doc)} pages")
                doc.close()
                return images

        for i in pages_to_process:
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Reduce resolution if image is too large
            if pix.width > 3000 or pix.height > 3000:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            images.append(image)

        doc.close()
        return images

    def _load_images(self, input_path: str, page_numbers: Optional[List[int]] = None) -> List[Union[str, Image.Image]]:
        """
        Load images from various input formats.

        Args:
            input_path (str): Path to image file, PDF file, or directory
            page_numbers (Optional[List[int]]): Specific page numbers to process for PDFs (1-indexed)

        Returns:
            List[Union[str, Image.Image]]: List of image paths or PIL Images
        """
        images = []

        if os.path.isdir(input_path):
            # Directory containing images or PDFs
            for file in sorted(os.listdir(input_path)):
                file_path = os.path.join(input_path, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(file_path)
                elif file.lower().endswith('.pdf'):
                    pdf_images = self._load_pdf_images(file_path, page_numbers=page_numbers)
                    images.extend(pdf_images)
        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Single image file
            images = [input_path]
        elif input_path.lower().endswith('.pdf'):
            # Single PDF file
            images = self._load_pdf_images(input_path, page_numbers=page_numbers)
        else:
            raise ValueError(f"Unsupported input format: {input_path}")

        return images

    def _colormap(self, N: int = 256, normalized: bool = False) -> np.ndarray:
        """Generate color map for visualization."""
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << (7 - j))
                g = g | (bitget(c, 1) << (7 - j))
                b = b | (bitget(c, 2) << (7 - j))
                c = c >> 3
            cmap[i] = np.array([r, g, b])

        if normalized:
            cmap = cmap.astype(np.float32) / 255.0

        return cmap

    def _visualize_results(self, image: Union[str, Image.Image], boxes: np.ndarray,
                          classes: np.ndarray, scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Visualize detection results on image.

        Args:
            image: Input image (path or PIL Image)
            boxes: Bounding boxes array
            classes: Class IDs array
            scores: Confidence scores array
            alpha: Transparency for overlay

        Returns:
            np.ndarray: Visualized image
        """
        # Load image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = cv2.imread(image)

        overlay = img_array.copy()
        cmap = self._colormap(N=len(self.id_to_names), normalized=False)

        # Draw bounding boxes
        for i, bbox in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, bbox)
            class_id = int(classes[i])
            class_name = self.id_to_names[class_id]

            text = f"{class_name}:{scores[i]:.3f}"
            color = tuple(int(c) for c in cmap[class_id])

            # Draw filled rectangle and border
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
            cv2.rectangle(img_array, (x_min, y_min), (x_max, y_max), color, 2)

            # Add text label
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(img_array, (x_min, y_min - text_height - baseline),
                         (x_min + text_width, y_min), color, -1)
            cv2.putText(img_array, text, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0, img_array)

        return img_array

    def predict(self, input_path: str, output_dir: str = "output",
               page_numbers: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Perform layout detection on input images or PDFs.

        Args:
            input_path (str): Path to input image, PDF, or directory
            output_dir (str): Directory to save results
            page_numbers (Optional[List[int]]): Specific page numbers to process for PDFs (1-indexed).
                                              If None, processes all pages. Ignored for image files.

        Returns:
            List[Dict[str, Any]]: List of detection results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load images
        images = self._load_images(input_path, page_numbers=page_numbers)
        results = []

        # Print info about page selection
        if page_numbers and input_path.lower().endswith('.pdf'):
            print(f"Processing specific pages: {page_numbers}")
        elif page_numbers and not input_path.lower().endswith('.pdf'):
            print("Warning: page_numbers parameter ignored for non-PDF input")

        for idx, image in enumerate(images):
            # Run inference
            result = self.model.predict(
                image,
                imgsz=self.img_size,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
                device=self.device
            )[0]

            # Extract detection results
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
            classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([])
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])

            # Apply additional NMS if needed
            if self.iou_thres > 0 and len(boxes) > 0:
                indices = self.nms_func(
                    boxes=torch.Tensor(boxes),
                    scores=torch.Tensor(scores),
                    iou_threshold=self.iou_thres
                )
                boxes = boxes[indices]
                scores = scores[indices]
                classes = classes[indices]

            # Generate base name for output files
            if isinstance(image, str):
                base_name = os.path.splitext(os.path.basename(image))[0]
            else:
                # For PDF pages, use actual page number if specific pages were selected
                if page_numbers and input_path.lower().endswith('.pdf'):
                    actual_page_num = page_numbers[idx]
                    base_name = f"pdf_page_{actual_page_num:04d}"
                else:
                    base_name = f"pdf_page_{idx+1:04d}"

            # Save visualization if enabled
            if self.visualize and len(boxes) > 0:
                vis_result = self._visualize_results(image, boxes, classes, scores)
                vis_path = os.path.join(output_dir, f"{base_name}_layout.png")
                cv2.imwrite(vis_path, vis_result)

            # Prepare detailed result dictionary
            detection_result = {
                "image_id": base_name,
                "image_info": {
                    "source": str(image) if isinstance(image, str) else f"pdf_page_{idx+1}",
                    "total_detections": len(boxes)
                },
                "detections": []
            }

            # Add individual detection details
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(classes[i])
                score = float(scores[i])
                cls_name = self.id_to_names[cls_id]

                detection = {
                    "bbox": {
                        "x1": float(box[0]),  # top-left x
                        "y1": float(box[1]),  # top-left y
                        "x2": float(box[2]),  # bottom-right x
                        "y2": float(box[3]),  # bottom-right y
                        "width": float(box[2] - box[0]),
                        "height": float(box[3] - box[1]),
                        "area": float((box[2] - box[0]) * (box[3] - box[1]))
                    },
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": score
                }
                detection_result["detections"].append(detection)

            # Add legacy format for backward compatibility
            detection_result["legacy_format"] = {
                "boxes": boxes.tolist(),
                "classes": classes.tolist(),
                "scores": scores.tolist(),
                "class_names": [self.id_to_names[int(cls)] for cls in classes]
            }

            results.append(detection_result)

        return results

    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get information about a PDF file.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict[str, Any]: PDF information including page count
        """
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("Input file is not a PDF")

        doc = fitz.open(pdf_path)
        info = {
            "file_path": pdf_path,
            "file_name": os.path.basename(pdf_path),
            "total_pages": len(doc),
            "page_numbers": list(range(1, len(doc) + 1))
        }
        doc.close()
        return info

    def save_results_json(self, results: List[Dict[str, Any]], output_dir: str,
                         format_type: str = "detailed") -> str:
        """
        Save detection results to JSON file.

        Args:
            results: Detection results from predict method
            output_dir: Directory to save JSON file
            format_type: Format type - "detailed", "simple", or "coco"

        Returns:
            str: Path to saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)

        if format_type == "detailed":
            # Enhanced format with all information
            json_path = os.path.join(output_dir, "detection_results_detailed.json")
            formatted_results = self._format_detailed_results(results)
        elif format_type == "simple":
            # Simple format with just essential information
            json_path = os.path.join(output_dir, "detection_results_simple.json")
            formatted_results = self._format_simple_results(results)
        elif format_type == "coco":
            # COCO-style format
            json_path = os.path.join(output_dir, "detection_results_coco.json")
            formatted_results = self._format_coco_results(results)
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")

        import json
        with open(json_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)

        return json_path

    def _format_detailed_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format results in detailed format."""
        return {
            "metadata": {
                "model_info": {
                    "model_path": self.model_path,
                    "device": self.device,
                    "confidence_threshold": self.conf_thres,
                    "iou_threshold": self.iou_thres,
                    "image_size": self.img_size
                },
                "class_mapping": self.id_to_names,
                "total_images": len(results),
                "total_detections": sum(len(r["detections"]) for r in results)
            },
            "results": results
        }

    def _format_simple_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format results in simple format."""
        simple_results = []
        for result in results:
            simple_result = {
                "image_id": result["image_id"],
                "detections": []
            }

            for detection in result["detections"]:
                simple_detection = {
                    "bbox": [
                        detection["bbox"]["x1"],
                        detection["bbox"]["y1"],
                        detection["bbox"]["x2"],
                        detection["bbox"]["y2"]
                    ],
                    "class": detection["class_name"],
                    "confidence": detection["confidence"]
                }
                simple_result["detections"].append(simple_detection)

            simple_results.append(simple_result)

        return simple_results

    def _format_coco_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format results in COCO-style format."""
        images = []
        annotations = []
        categories = []

        # Create categories
        for class_id, class_name in self.id_to_names.items():
            categories.append({
                "id": class_id,
                "name": class_name,
                "supercategory": "layout_element"
            })

        annotation_id = 1

        for img_idx, result in enumerate(results):
            # Add image info
            images.append({
                "id": img_idx + 1,
                "file_name": result["image_id"],
                "width": 0,  # Would need actual image dimensions
                "height": 0  # Would need actual image dimensions
            })

            # Add annotations
            for detection in result["detections"]:
                bbox = detection["bbox"]
                annotations.append({
                    "id": annotation_id,
                    "image_id": img_idx + 1,
                    "category_id": detection["class_id"],
                    "bbox": [bbox["x1"], bbox["y1"], bbox["width"], bbox["height"]],
                    "area": bbox["area"],
                    "iscrowd": 0,
                    "score": detection["confidence"]
                })
                annotation_id += 1

        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }


def main():
    """Example usage of the LayoutDetector."""
    # Example configuration
    model_path = "models/Layout/YOLO/doclayout_yolo_ft.pt"  # Update this path

    # Initialize detector
    detector = LayoutDetector(
        model_path=model_path,
        device='cpu',  # Change to 'cuda' if GPU is available
        conf_thres=0.25,
        iou_thres=0.45,
        img_size=1024,
        visualize=True
    )

    # Example usage
    input_path = "input_images"  # Can be image file, PDF file, or directory
    output_dir = "output_results"

    try:
        results = detector.predict(input_path, output_dir)
        print(f"Processed {len(results)} images/pages")
        print(f"Results saved to: {output_dir}")

        # Print summary of detections
        for result in results:
            print(f"\nImage: {result['image_id']}")
            print(f"Detected {len(result['boxes'])} layout elements:")
            for i, class_name in enumerate(result['class_names']):
                print(f"  - {class_name}: {result['scores'][i]:.3f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
