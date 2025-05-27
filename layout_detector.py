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
    
    def _load_pdf_images(self, pdf_path: str, dpi: int = 144) -> List[Image.Image]:
        """
        Load images from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            dpi (int): DPI for rendering PDF pages
            
        Returns:
            List[Image.Image]: List of PIL Images from PDF pages
        """
        images = []
        doc = fitz.open(pdf_path)
        
        for i in range(len(doc)):
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
    
    def _load_images(self, input_path: str) -> List[Union[str, Image.Image]]:
        """
        Load images from various input formats.
        
        Args:
            input_path (str): Path to image file, PDF file, or directory
            
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
                    pdf_images = self._load_pdf_images(file_path)
                    images.extend(pdf_images)
        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Single image file
            images = [input_path]
        elif input_path.lower().endswith('.pdf'):
            # Single PDF file
            images = self._load_pdf_images(input_path)
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
    
    def predict(self, input_path: str, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        Perform layout detection on input images or PDFs.
        
        Args:
            input_path (str): Path to input image, PDF, or directory
            output_dir (str): Directory to save results
            
        Returns:
            List[Dict[str, Any]]: List of detection results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        images = self._load_images(input_path)
        results = []
        
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
                base_name = f"pdf_page_{idx+1:04d}"
            
            # Save visualization if enabled
            if self.visualize and len(boxes) > 0:
                vis_result = self._visualize_results(image, boxes, classes, scores)
                vis_path = os.path.join(output_dir, f"{base_name}_layout.png")
                cv2.imwrite(vis_path, vis_result)
            
            # Prepare result dictionary
            detection_result = {
                "image_id": base_name,
                "boxes": boxes.tolist(),
                "classes": classes.tolist(),
                "scores": scores.tolist(),
                "class_names": [self.id_to_names[int(cls)] for cls in classes]
            }
            
            results.append(detection_result)
        
        return results


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
