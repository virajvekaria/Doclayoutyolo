#!/usr/bin/env python3
"""
Simple script to run layout detection using configuration file.
"""

import argparse
import yaml
import os
from layout_detector import LayoutDetector


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_page_numbers(page_str: str) -> list:
    """
    Parse page number string into a list of integers.

    Args:
        page_str (str): Page specification like '1,3,5' or '1-5' or '1,3-7,10'

    Returns:
        list: List of page numbers
    """
    if not page_str:
        return None

    pages = []
    parts = page_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like '3-7'
            start, end = part.split('-', 1)
            start, end = int(start.strip()), int(end.strip())
            pages.extend(range(start, end + 1))
        else:
            # Handle single page
            pages.append(int(part))

    return sorted(list(set(pages)))  # Remove duplicates and sort


def main():
    parser = argparse.ArgumentParser(description="Run DocLayout-YOLO Layout Detection")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input path (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model path (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["detailed", "simple", "coco"],
        default="detailed",
        help="JSON output format (default: detailed)"
    )
    parser.add_argument(
        "--pages",
        type=str,
        help="Specific page numbers to process for PDFs (e.g., '1,3,5' or '1-5' or '1,3-7,10')"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show PDF information (page count) and exit"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments if provided
    if args.input:
        config['paths']['input'] = args.input
    if args.output:
        config['paths']['output'] = args.output
    if args.model:
        config['model']['model_path'] = args.model
    if args.device:
        config['model']['device'] = args.device

    # Parse page numbers
    page_numbers = parse_page_numbers(args.pages) if args.pages else None

    # Initialize detector
    print("Initializing DocLayout-YOLO detector...")
    detector = LayoutDetector(
        model_path=config['model']['model_path'],
        device=config['model']['device'],
        conf_thres=config['model']['conf_thres'],
        iou_thres=config['model']['iou_thres'],
        img_size=config['model']['img_size'],
        visualize=config['visualization']['enabled']
    )

    # Get input and output paths
    input_path = config['paths']['input']
    output_dir = config['paths']['output']

    # Handle --info flag
    if args.info:
        if input_path.lower().endswith('.pdf'):
            try:
                pdf_info = detector.get_pdf_info(input_path)
                print(f"PDF Information:")
                print(f"  File: {pdf_info['file_name']}")
                print(f"  Total pages: {pdf_info['total_pages']}")
                print(f"  Available pages: {', '.join(map(str, pdf_info['page_numbers']))}")
                return 0
            except Exception as e:
                print(f"Error getting PDF info: {e}")
                return 1
        else:
            print("--info flag only works with PDF files")
            return 1

    print(f"Processing input: {input_path}")
    print(f"Output directory: {output_dir}")

    if page_numbers:
        print(f"Selected pages: {page_numbers}")

    try:
        results = detector.predict(input_path, output_dir, page_numbers=page_numbers)

        # Save results to JSON in specified format
        results_file = detector.save_results_json(results, output_dir, args.format)

        # Also save in simple format for backward compatibility
        if args.format != "simple":
            detector.save_results_json(results, output_dir, "simple")

        print(f"\nProcessing completed!")
        print(f"Processed {len(results)} images/pages")
        print(f"Results saved to: {output_dir}")
        print(f"Detailed results: {results_file}")

        # Print summary
        total_detections = sum(len(result['detections']) for result in results)
        print(f"Total layout elements detected: {total_detections}")

        # Count detections by class
        class_counts = {}
        for result in results:
            for detection in result['detections']:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if class_counts:
            print("\nDetection summary by class:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}")

    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
