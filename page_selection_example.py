#!/usr/bin/env python3
"""
Example script demonstrating page selection functionality for PDF processing.
"""

import os
from layout_detector import LayoutDetector


def demonstrate_page_selection():
    """Demonstrate different ways to select pages from PDFs."""
    print("=== Page Selection Functionality Demo ===\n")
    
    # Initialize detector
    detector = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",  # Update this path
        device='cpu',
        visualize=True
    )
    
    pdf_path = "sample_document.pdf"  # Update this path
    
    # Check if PDF exists and get info
    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file to run this demo.")
        return
    
    try:
        # Get PDF information
        pdf_info = detector.get_pdf_info(pdf_path)
        print(f"📄 PDF Information:")
        print(f"   File: {pdf_info['file_name']}")
        print(f"   Total pages: {pdf_info['total_pages']}")
        print(f"   Available pages: {', '.join(map(str, pdf_info['page_numbers']))}")
        print()
        
        # Example 1: Process all pages (default behavior)
        print("1️⃣  Processing ALL pages (default):")
        print("-" * 40)
        results_all = detector.predict(pdf_path, "output_all_pages")
        print(f"   ✅ Processed {len(results_all)} pages")
        print(f"   📁 Results saved to: output_all_pages/")
        print()
        
        # Example 2: Process single page
        print("2️⃣  Processing SINGLE page (page 1):")
        print("-" * 40)
        results_single = detector.predict(pdf_path, "output_page_1", page_numbers=[1])
        print(f"   ✅ Processed {len(results_single)} page(s)")
        print(f"   📁 Results saved to: output_page_1/")
        print()
        
        # Example 3: Process specific pages
        if pdf_info['total_pages'] >= 3:
            print("3️⃣  Processing SPECIFIC pages (pages 1, 3):")
            print("-" * 40)
            results_specific = detector.predict(pdf_path, "output_pages_1_3", page_numbers=[1, 3])
            print(f"   ✅ Processed {len(results_specific)} page(s)")
            print(f"   📁 Results saved to: output_pages_1_3/")
            print()
        
        # Example 4: Process page range
        if pdf_info['total_pages'] >= 5:
            print("4️⃣  Processing PAGE RANGE (pages 2-4):")
            print("-" * 40)
            page_range = list(range(2, 5))  # Pages 2, 3, 4
            results_range = detector.predict(pdf_path, "output_pages_2_4", page_numbers=page_range)
            print(f"   ✅ Processed {len(results_range)} page(s)")
            print(f"   📁 Results saved to: output_pages_2_4/")
            print()
        
        # Example 5: Process last page
        last_page = pdf_info['total_pages']
        print(f"5️⃣  Processing LAST page (page {last_page}):")
        print("-" * 40)
        results_last = detector.predict(pdf_path, "output_last_page", page_numbers=[last_page])
        print(f"   ✅ Processed {len(results_last)} page(s)")
        print(f"   📁 Results saved to: output_last_page/")
        print()
        
        # Show detection statistics
        print("📊 Detection Statistics:")
        print("-" * 40)
        
        def show_stats(results, description):
            total_detections = sum(len(result['detections']) for result in results)
            print(f"   {description}:")
            print(f"     Pages processed: {len(results)}")
            print(f"     Total detections: {total_detections}")
            if results:
                avg_detections = total_detections / len(results)
                print(f"     Avg detections per page: {avg_detections:.1f}")
            print()
        
        show_stats(results_all, "All pages")
        show_stats(results_single, "Single page (1)")
        if pdf_info['total_pages'] >= 3:
            show_stats(results_specific, "Specific pages (1,3)")
        show_stats(results_last, f"Last page ({last_page})")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_command_line_usage():
    """Show command line usage examples."""
    print("\n=== Command Line Usage Examples ===")
    print()
    
    print("📋 Basic Commands:")
    print("-" * 50)
    print("# Get PDF information")
    print("python run_detection.py --input document.pdf --info")
    print()
    print("# Process all pages (default)")
    print("python run_detection.py --input document.pdf --output results")
    print()
    print("# Process single page")
    print("python run_detection.py --input document.pdf --output results --pages 1")
    print()
    print("# Process specific pages")
    print("python run_detection.py --input document.pdf --output results --pages '1,3,5'")
    print()
    print("# Process page range")
    print("python run_detection.py --input document.pdf --output results --pages '2-5'")
    print()
    print("# Process mixed selection")
    print("python run_detection.py --input document.pdf --output results --pages '1,3-7,10'")
    print()
    
    print("🎯 Advanced Examples:")
    print("-" * 50)
    print("# Process specific pages with GPU and simple JSON format")
    print("python run_detection.py --input document.pdf --output results \\")
    print("                       --pages '1,5,10' --device cuda --format simple")
    print()
    print("# Process page range with custom model")
    print("python run_detection.py --input document.pdf --output results \\")
    print("                       --pages '1-10' --model custom_model.pt")
    print()


def demonstrate_error_handling():
    """Demonstrate error handling for invalid page selections."""
    print("\n=== Error Handling Examples ===")
    print()
    
    detector = LayoutDetector(
        model_path="models/doclayout_yolo_ft.pt",
        device='cpu',
        visualize=False
    )
    
    pdf_path = "sample_document.pdf"
    
    if not os.path.exists(pdf_path):
        print("⚠️  PDF file not found for error handling demo")
        return
    
    try:
        pdf_info = detector.get_pdf_info(pdf_path)
        total_pages = pdf_info['total_pages']
        
        print(f"📄 Testing with PDF that has {total_pages} pages:")
        print()
        
        # Test invalid page numbers
        test_cases = [
            ([0], "Page 0 (invalid - pages start from 1)"),
            ([total_pages + 1], f"Page {total_pages + 1} (beyond last page)"),
            ([-1], "Negative page number"),
            ([1, total_pages + 5], f"Mix of valid and invalid pages")
        ]
        
        for pages, description in test_cases:
            print(f"🧪 Testing: {description}")
            try:
                results = detector.predict(pdf_path, "output_test", page_numbers=pages)
                if results:
                    print(f"   ✅ Processed {len(results)} valid page(s)")
                else:
                    print(f"   ⚠️  No valid pages found")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            print()
            
    except Exception as e:
        print(f"❌ Error in error handling demo: {e}")


def main():
    """Run all demonstrations."""
    print("DocLayout-YOLO Page Selection Functionality")
    print("=" * 60)
    
    demonstrate_page_selection()
    demonstrate_command_line_usage()
    demonstrate_error_handling()
    
    print("\n" + "=" * 60)
    print("✨ Page Selection Demo Complete!")
    print()
    print("💡 Tips:")
    print("- Use --info to check PDF page count before processing")
    print("- Page numbers are 1-indexed (first page is 1, not 0)")
    print("- Invalid page numbers are automatically filtered out")
    print("- Processing specific pages saves time and storage space")
    print("- Visualization files will be named with actual page numbers")


if __name__ == "__main__":
    main()
