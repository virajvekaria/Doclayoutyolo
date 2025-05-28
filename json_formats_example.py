#!/usr/bin/env python3
"""
Example script demonstrating different JSON output formats for layout detection results.
"""

import json
import os
from layout_detector import LayoutDetector


def demonstrate_json_formats():
    """Demonstrate different JSON output formats."""
    print("=== JSON Output Formats Demonstration ===\n")
    
    # Initialize detector
    detector = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",  # Update this path
        device='cpu',
        conf_thres=0.25,
        visualize=False  # Disable visualization for this demo
    )
    
    try:
        # Run detection on sample input
        results = detector.predict("sample_image.jpg", "output_json_demo")
        
        if not results:
            print("No results to demonstrate. Please provide a valid input image.")
            return
        
        # Save and demonstrate different formats
        output_dir = "output_json_demo"
        
        # 1. Detailed Format
        print("1. DETAILED FORMAT")
        print("-" * 50)
        detailed_file = detector.save_results_json(results, output_dir, "detailed")
        print(f"Saved to: {detailed_file}")
        
        with open(detailed_file, 'r') as f:
            detailed_data = json.load(f)
        
        print("Structure:")
        print("- metadata: Model info, class mapping, statistics")
        print("- results: Array of detection results per image")
        print("  - image_id: Identifier for the image")
        print("  - image_info: Source and detection count")
        print("  - detections: Array of individual detections")
        print("    - bbox: Detailed bounding box info (x1,y1,x2,y2,width,height,area)")
        print("    - class_id: Numeric class identifier")
        print("    - class_name: Human-readable class name")
        print("    - confidence: Detection confidence score")
        print("  - legacy_format: Backward compatibility format")
        
        print(f"\nSample (first detection):")
        if detailed_data["results"] and detailed_data["results"][0]["detections"]:
            first_detection = detailed_data["results"][0]["detections"][0]
            print(json.dumps(first_detection, indent=2))
        
        # 2. Simple Format
        print("\n\n2. SIMPLE FORMAT")
        print("-" * 50)
        simple_file = detector.save_results_json(results, output_dir, "simple")
        print(f"Saved to: {simple_file}")
        
        with open(simple_file, 'r') as f:
            simple_data = json.load(f)
        
        print("Structure:")
        print("- Array of results per image")
        print("  - image_id: Identifier for the image")
        print("  - detections: Array of detections")
        print("    - bbox: [x1, y1, x2, y2] coordinates")
        print("    - class: Class name")
        print("    - confidence: Detection confidence")
        
        print(f"\nSample (first detection):")
        if simple_data and simple_data[0]["detections"]:
            first_detection = simple_data[0]["detections"][0]
            print(json.dumps(first_detection, indent=2))
        
        # 3. COCO Format
        print("\n\n3. COCO FORMAT")
        print("-" * 50)
        coco_file = detector.save_results_json(results, output_dir, "coco")
        print(f"Saved to: {coco_file}")
        
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        print("Structure:")
        print("- images: Array of image information")
        print("- annotations: Array of detection annotations")
        print("  - bbox: [x, y, width, height] format")
        print("  - category_id: Numeric class ID")
        print("  - score: Detection confidence")
        print("- categories: Array of class definitions")
        
        print(f"\nSample annotation:")
        if coco_data["annotations"]:
            first_annotation = coco_data["annotations"][0]
            print(json.dumps(first_annotation, indent=2))
        
        print(f"\nSample category:")
        if coco_data["categories"]:
            first_category = coco_data["categories"][0]
            print(json.dumps(first_category, indent=2))
        
        # Statistics
        print("\n\n=== STATISTICS ===")
        print(f"Total images processed: {len(results)}")
        total_detections = sum(len(result['detections']) for result in results)
        print(f"Total detections: {total_detections}")
        
        # File sizes
        detailed_size = os.path.getsize(detailed_file) / 1024
        simple_size = os.path.getsize(simple_file) / 1024
        coco_size = os.path.getsize(coco_file) / 1024
        
        print(f"\nFile sizes:")
        print(f"- Detailed format: {detailed_size:.1f} KB")
        print(f"- Simple format: {simple_size:.1f} KB")
        print(f"- COCO format: {coco_size:.1f} KB")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A valid model file")
        print("2. A sample input image")


def show_format_comparison():
    """Show a side-by-side comparison of the same detection in different formats."""
    print("\n=== FORMAT COMPARISON ===")
    print("Same detection represented in different formats:\n")
    
    # Example detection data
    sample_detection = {
        "bbox": {"x1": 100.0, "y1": 50.0, "x2": 300.0, "y2": 150.0, "width": 200.0, "height": 100.0, "area": 20000.0},
        "class_id": 0,
        "class_name": "title",
        "confidence": 0.95
    }
    
    print("DETAILED FORMAT:")
    print(json.dumps(sample_detection, indent=2))
    
    print("\nSIMPLE FORMAT:")
    simple_detection = {
        "bbox": [100.0, 50.0, 300.0, 150.0],
        "class": "title",
        "confidence": 0.95
    }
    print(json.dumps(simple_detection, indent=2))
    
    print("\nCOCO FORMAT:")
    coco_annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 0,
        "bbox": [100.0, 50.0, 200.0, 100.0],  # [x, y, width, height]
        "area": 20000.0,
        "iscrowd": 0,
        "score": 0.95
    }
    print(json.dumps(coco_annotation, indent=2))


def usage_recommendations():
    """Provide recommendations for when to use each format."""
    print("\n=== USAGE RECOMMENDATIONS ===")
    
    print("\n📊 DETAILED FORMAT:")
    print("✅ Best for: Comprehensive analysis, debugging, research")
    print("✅ Contains: All available information, metadata, statistics")
    print("✅ Use when: You need complete information about the detection process")
    print("❌ Drawbacks: Larger file size, more complex structure")
    
    print("\n🎯 SIMPLE FORMAT:")
    print("✅ Best for: Integration with other systems, quick processing")
    print("✅ Contains: Essential detection information only")
    print("✅ Use when: You need clean, minimal data for downstream processing")
    print("❌ Drawbacks: Less metadata, no model information")
    
    print("\n🏷️ COCO FORMAT:")
    print("✅ Best for: Computer vision research, model evaluation")
    print("✅ Contains: Standard format compatible with COCO evaluation tools")
    print("✅ Use when: You need compatibility with existing CV frameworks")
    print("❌ Drawbacks: More complex structure, requires understanding of COCO format")


def main():
    """Main function to run all demonstrations."""
    demonstrate_json_formats()
    show_format_comparison()
    usage_recommendations()
    
    print("\n" + "=" * 60)
    print("JSON Formats Demonstration Complete!")
    print("Check the 'output_json_demo' directory for example files.")


if __name__ == "__main__":
    main()
