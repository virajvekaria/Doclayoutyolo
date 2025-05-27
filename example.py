#!/usr/bin/env python3
"""
Example script demonstrating how to use the DocLayout-YOLO Layout Detection Pipeline.
"""

import os
import json
from layout_detector import LayoutDetector


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")

    # Initialize detector
    detector = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",  # Update this path
        device='cpu',  # Change to 'cuda' if GPU available
        conf_thres=0.25,
        iou_thres=0.45,
        img_size=1024,
        visualize=True
    )

    # Process a single image
    try:
        results = detector.predict("sample_image.jpg", "output_basic")
        print(f"Processed {len(results)} images")

        # Print results
        for result in results:
            print(f"\nImage: {result['image_id']}")
            print(f"Detected {len(result['boxes'])} layout elements:")
            for i, class_name in enumerate(result['class_names']):
                box = result['boxes'][i]
                score = result['scores'][i]
                print(f"  - {class_name}: {score:.3f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Downloaded the model file")
        print("2. Updated the model_path")
        print("3. Provided a valid input image")


def example_pdf_processing():
    """Example of processing PDF files."""
    print("\n=== PDF Processing Example ===")

    detector = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",
        device='cpu',
        visualize=True
    )

    try:
        # Get PDF information first
        pdf_path = "sample_document.pdf"
        pdf_info = detector.get_pdf_info(pdf_path)
        print(f"PDF Info: {pdf_info['total_pages']} pages")

        # Process all pages
        results_all = detector.predict(pdf_path, "output_pdf_all")
        print(f"Processed all {len(results_all)} pages")

        # Process specific pages (e.g., first 3 pages)
        if pdf_info['total_pages'] >= 3:
            results_specific = detector.predict(pdf_path, "output_pdf_specific", page_numbers=[1, 2, 3])
            print(f"Processed specific pages: {len(results_specific)} pages")

        # Analyze results by page
        for result in results_all:
            print(f"\nPage ({result['image_id']}):")

            # Count elements by type
            element_counts = {}
            for detection in result['detections']:
                class_name = detection['class_name']
                element_counts[class_name] = element_counts.get(class_name, 0) + 1

            for element_type, count in element_counts.items():
                print(f"  {element_type}: {count}")

    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example of batch processing multiple files."""
    print("\n=== Batch Processing Example ===")

    detector = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",
        device='cpu',
        conf_thres=0.3,  # Higher confidence threshold
        visualize=True
    )

    try:
        # Process all files in a directory
        results = detector.predict("input_directory", "output_batch")

        print(f"Processed {len(results)} files")

        # Generate summary statistics
        total_elements = sum(len(result['boxes']) for result in results)
        print(f"Total layout elements detected: {total_elements}")

        # Count by element type across all files
        global_counts = {}
        for result in results:
            for class_name in result['class_names']:
                global_counts[class_name] = global_counts.get(class_name, 0) + 1

        print("\nGlobal element distribution:")
        for element_type, count in sorted(global_counts.items()):
            percentage = (count / total_elements) * 100 if total_elements > 0 else 0
            print(f"  {element_type}: {count} ({percentage:.1f}%)")

    except Exception as e:
        print(f"Error: {e}")


def example_page_selection():
    """Example of page selection functionality."""
    print("\n=== Page Selection Example ===")

    detector = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",
        device='cpu',
        visualize=True
    )

    try:
        pdf_path = "sample_document.pdf"

        # Get PDF information
        pdf_info = detector.get_pdf_info(pdf_path)
        print(f"PDF has {pdf_info['total_pages']} pages")

        # Example 1: Process single page
        print("\n1. Processing single page (page 1):")
        results_single = detector.predict(pdf_path, "output_page1", page_numbers=[1])
        print(f"   Processed {len(results_single)} page")

        # Example 2: Process specific pages
        if pdf_info['total_pages'] >= 3:
            print("\n2. Processing specific pages (1, 3):")
            results_specific = detector.predict(pdf_path, "output_pages_1_3", page_numbers=[1, 3])
            print(f"   Processed {len(results_specific)} pages")

        # Example 3: Process page range
        if pdf_info['total_pages'] >= 5:
            print("\n3. Processing page range (2-4):")
            page_range = list(range(2, 5))  # Pages 2, 3, 4
            results_range = detector.predict(pdf_path, "output_pages_2_4", page_numbers=page_range)
            print(f"   Processed {len(results_range)} pages")

        # Example 4: Process last page
        last_page = pdf_info['total_pages']
        print(f"\n4. Processing last page ({last_page}):")
        results_last = detector.predict(pdf_path, "output_last_page", page_numbers=[last_page])
        print(f"   Processed {len(results_last)} page")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a valid PDF file for this example")


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")

    # High precision configuration
    detector_precise = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",
        device='cpu',
        conf_thres=0.5,   # Higher confidence threshold
        iou_thres=0.3,    # Lower IoU threshold (more strict NMS)
        img_size=1280,    # Larger input size for better accuracy
        visualize=True
    )

    # Fast processing configuration
    detector_fast = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",
        device='cpu',
        conf_thres=0.2,   # Lower confidence threshold
        iou_thres=0.6,    # Higher IoU threshold (less strict NMS)
        img_size=640,     # Smaller input size for faster processing
        visualize=False   # Disable visualization for speed
    )

    print("Configurations created:")
    print("- Precise: High accuracy, slower processing")
    print("- Fast: Lower accuracy, faster processing")


def example_result_analysis():
    """Example of analyzing detection results."""
    print("\n=== Result Analysis Example ===")

    # Load results from a previous run
    results_file = "output_basic/detection_results.json"

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)

        print(f"Analyzing results from {results_file}")

        for result in results:
            print(f"\nImage: {result['image_id']}")

            # Find the largest element (by area)
            if result['boxes']:
                areas = []
                for box in result['boxes']:
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    areas.append(width * height)

                largest_idx = areas.index(max(areas))
                largest_element = result['class_names'][largest_idx]
                largest_score = result['scores'][largest_idx]

                print(f"Largest element: {largest_element} (score: {largest_score:.3f})")

            # Find high-confidence detections
            high_conf_elements = [
                (name, score) for name, score in zip(result['class_names'], result['scores'])
                if score > 0.8
            ]

            if high_conf_elements:
                print(f"High-confidence detections ({len(high_conf_elements)}):")
                for name, score in high_conf_elements:
                    print(f"  - {name}: {score:.3f}")
    else:
        print(f"Results file not found: {results_file}")
        print("Run a detection first to generate results.")


def main():
    """Run all examples."""
    print("DocLayout-YOLO Layout Detection Pipeline Examples")
    print("=" * 50)

    # Check if model exists
    model_path = "models/doclayout_yolo_ft.pt"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("Please download the model from:")
        print("https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/doclayout_yolo_ft.pt")
        print("\nYou can still run the examples by updating the model_path in each function.")
        print()

    # Run examples
    example_basic_usage()
    example_pdf_processing()
    example_page_selection()
    example_batch_processing()
    example_custom_configuration()
    example_result_analysis()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Check the output directories for results.")


if __name__ == "__main__":
    main()
