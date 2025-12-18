#!/usr/bin/env python3
"""
Bitwise Video Change Detection
Creates a binary mask video showing pixel changes over a rolling frame range.
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


def create_change_mask_video(input_path, output_path, frame_range=1, threshold=10, blur=0):
    """
    Create a binary mask video showing changes between frames.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        frame_range: Number of frames back to compare against (default: 1 for consecutive frames)
        threshold: Pixel difference threshold before considering it changed (default: 10)
        blur: Gaussian blur kernel size for smoothing (must be odd, 0 = no blur, default: 0)
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} @ {fps} fps, {total_frames} frames")
    print(f"Frame comparison range: {frame_range} frames back")
    print(f"Threshold: {threshold}")
    print(f"Blur kernel size: {blur if blur > 0 else 'none'}")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        sys.exit(1)
    
    # Store frames in a rolling buffer
    frame_buffer = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Add to buffer
        frame_buffer.append(gray_frame)
        
        # Create mask
        if len(frame_buffer) > frame_range:
            # Compare current frame with frame N positions back
            reference_frame = frame_buffer[-frame_range - 1]
            current_frame = frame_buffer[-1]
            
            # Calculate absolute difference
            diff = cv2.absdiff(current_frame, reference_frame)
            
            # Apply threshold to create binary mask
            _, binary_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Optional: Apply XOR operation for cleaner bitwise difference
            # This highlights pixels that differ significantly
            xor_mask = cv2.bitwise_xor(
                cv2.threshold(current_frame, 127, 255, cv2.THRESH_BINARY)[1],
                cv2.threshold(reference_frame, 127, 255, cv2.THRESH_BINARY)[1]
            )
            
            # Combine threshold-based and XOR-based masks
            # Use threshold-based as it's more sensitive to subtle changes
            final_mask = binary_mask
            
            # Apply Gaussian blur if specified
            if blur > 0:
                final_mask = cv2.GaussianBlur(final_mask, (blur, blur), 0)
            
            out.write(final_mask)
            
            # Keep buffer size manageable
            if len(frame_buffer) > frame_range + 1:
                frame_buffer.pop(0)
        else:
            # For first N frames, output black (no comparison possible)
            black_frame = np.zeros((height, width), dtype=np.uint8)
            out.write(black_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"\nDone! Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Create binary mask video showing pixel changes over time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare consecutive frames (1 frame back)
  python bitwise_video_change.py input.mp4 output.mp4
  
  # Compare with 5 frames back
  python bitwise_video_change.py input.mp4 output.mp4 --range 5
  
  # Use custom threshold for change detection
  python bitwise_video_change.py input.mp4 output.mp4 --range 3 --threshold 20
  
  # Add blur to smooth the mask (kernel size must be odd)
  python bitwise_video_change.py input.mp4 output.mp4 --blur 5
  
  # Combine all parameters
  python bitwise_video_change.py input.mp4 output.mp4 --range 5 --threshold 15 --blur 7
        """
    )
    
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('output', help='Path to output video file')
    parser.add_argument('--range', '-r', type=int, default=1,
                        help='Number of frames back to compare (default: 1)')
    parser.add_argument('--threshold', '-t', type=int, default=10,
                        help='Pixel difference threshold (0-255, default: 10)')
    parser.add_argument('--blur', '-b', type=int, default=0,
                        help='Gaussian blur kernel size for smoothing (must be odd, 0 = no blur, default: 0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    if args.range < 1:
        print("Error: Frame range must be at least 1")
        sys.exit(1)
    
    if not 0 <= args.threshold <= 255:
        print("Error: Threshold must be between 0 and 255")
        sys.exit(1)
    
    if args.blur < 0:
        print("Error: Blur kernel size must be non-negative")
        sys.exit(1)
    
    if args.blur > 0 and args.blur % 2 == 0:
        print("Error: Blur kernel size must be odd (e.g., 3, 5, 7, 9...)")
        sys.exit(1)
    
    create_change_mask_video(args.input, args.output, args.range, args.threshold, args.blur)


if __name__ == '__main__':
    main()
