#!/usr/bin/env python3
"""
Simple script to run layout detection using configuration file.
"""

import argparse
import yaml
import json
import os
from layout_detector import LayoutDetector


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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
    
    # Run detection
    input_path = config['paths']['input']
    output_dir = config['paths']['output']
    
    print(f"Processing input: {input_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        results = detector.predict(input_path, output_dir)
        
        # Save results to JSON
        results_file = os.path.join(output_dir, "detection_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessing completed!")
        print(f"Processed {len(results)} images/pages")
        print(f"Results saved to: {output_dir}")
        print(f"Detailed results: {results_file}")
        
        # Print summary
        total_detections = sum(len(result['boxes']) for result in results)
        print(f"Total layout elements detected: {total_detections}")
        
        # Count detections by class
        class_counts = {}
        for result in results:
            for class_name in result['class_names']:
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
