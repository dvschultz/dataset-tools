#!/usr/bin/env python3
"""
Simple Video Resizer
Resize videos to a specified height while maintaining aspect ratio.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def resize_video(input_path, output_path, height=480, crf=23, preset='medium', codec='libx264'):
    """
    Resize a video to specified height.

    Args:
        input_path: Source video path
        output_path: Destination video path
        height: Target height in pixels
        crf: Quality (18=high, 23=good, 28=lower)
        preset: Encoding speed (ultrafast, fast, medium, slow)
        codec: Video codec (libx264, libx265, etc.)

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', f'scale=-1:{height}',
        '-c:v', codec,
        '-crf', str(crf),
        '-preset', preset,
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-y',
        str(output_path)
    ]

    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    if result.returncode != 0:
        error_msg = result.stderr.decode('utf-8', errors='ignore')
        print(f"Error resizing {input_path.name}: {error_msg[-200:]}", file=sys.stderr)
        return False

    return True


def process_single_video(input_path, output_path, height, crf, preset, codec, verbose):
    """Process a single video file."""
    input_path = Path(input_path)

    if output_path:
        output_path = Path(output_path)
    else:
        # Default: same name in current directory
        output_path = Path(f"{input_path.stem}_resized{input_path.suffix}")

    if verbose:
        print(f"Resizing: {input_path.name}")
        print(f"  Output: {output_path}")
        print(f"  Height: {height}px")
        print(f"  Quality (CRF): {crf}")

    success = resize_video(input_path, output_path, height, crf, preset, codec)

    if success:
        if verbose:
            print(f"  ✓ Complete: {output_path}")
        return True
    else:
        print(f"  ✗ Failed: {input_path.name}", file=sys.stderr)
        return False


def process_directory(input_dir, output_dir, height, crf, preset, codec, verbose):
    """Process all videos in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(ext))
        video_files.extend(input_dir.glob(ext.upper()))

    video_files = sorted(set(video_files))

    # Filter out hidden files that start with a dot
    video_files = [f for f in video_files if not f.name.startswith('.')]

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos to resize")
    print(f"Output directory: {output_dir}")
    print(f"Target height: {height}px")
    print(f"Quality (CRF): {crf}")
    print("="*60)

    success_count = 0

    for i, video_path in enumerate(video_files, 1):
        output_path = output_dir / video_path.name

        if verbose:
            print(f"\n[{i}/{len(video_files)}] {video_path.name}")
        else:
            print(f"[{i}/{len(video_files)}] Processing {video_path.name}...", end=' ')

        success = resize_video(video_path, output_path, height, crf, preset, codec)

        if success:
            success_count += 1
            if not verbose:
                print("✓")
        else:
            if not verbose:
                print("✗")

    print("\n" + "="*60)
    print(f"Complete: {success_count}/{len(video_files)} videos resized successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Resize videos to specified height while maintaining aspect ratio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize single video to 480p
  %(prog)s -i video.mp4 -o video_480p.mp4 --height 480

  # Resize to 720p with high quality
  %(prog)s -i video.mp4 -o video_720p.mp4 --height 720 --crf 18

  # Resize entire directory to 480p
  %(prog)s -d videos/ -o videos_480p/ --height 480

  # Fast encoding for quick previews
  %(prog)s -d videos/ -o previews/ --height 360 --preset ultrafast

  # High quality 1080p with H.265
  %(prog)s -i video.mp4 -o video_1080p.mp4 --height 1080 --codec libx265 --crf 18

Quality Guidelines:
  CRF Values (lower = higher quality, larger file):
    18-22: Very high quality (archival)
    23-28: Good quality (default: 23)
    28-32: Lower quality (small files)

  Presets (speed vs compression):
    ultrafast: Fastest encoding, larger files
    fast:      Fast encoding, good compression
    medium:    Balanced (default)
    slow:      Better compression, slower
    veryslow:  Best compression, very slow
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input',
                            help='Single video file to resize')
    input_group.add_argument('-d', '--directory',
                            help='Directory containing videos to resize')

    # Output options
    parser.add_argument('-o', '--output',
                       help='Output file (for -i) or directory (for -d)')

    # Resize options
    parser.add_argument('--height', type=int, default=480,
                       help='Target height in pixels (default: 480)')
    parser.add_argument('--crf', type=int, default=23,
                       help='Quality: 18=high, 23=good, 28=lower (default: 23)')
    parser.add_argument('--preset', default='medium',
                       choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                               'medium', 'slow', 'slower', 'veryslow'],
                       help='Encoding speed preset (default: medium)')
    parser.add_argument('--codec', default='libx264',
                       choices=['libx264', 'libx265', 'libvpx-vp9'],
                       help='Video codec (default: libx264)')

    # General options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed progress information')

    args = parser.parse_args()

    # Validation
    if args.directory and not args.output:
        parser.error('-d/--directory requires -o/--output')

    # Process
    if args.input:
        process_single_video(
            args.input,
            args.output,
            args.height,
            args.crf,
            args.preset,
            args.codec,
            args.verbose
        )
    else:
        process_directory(
            args.directory,
            args.output,
            args.height,
            args.crf,
            args.preset,
            args.codec,
            args.verbose
        )


if __name__ == '__main__':
    main()
