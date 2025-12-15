#!/usr/bin/env python3
"""
8mm Scene Detection and Clip Extraction
Detect scenes on low-res video, apply cuts to high-res originals.
Supports both black & white and color footage with optimized settings.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import json


def detect_scenes_get_timestamps(video_path, 
                                 threshold=27, 
                                 min_duration=2.0,
                                 footage_type='color',
                                 adaptive_threshold=False):
    """
    Detect scenes on video, return precise timestamps.
    
    Args:
        video_path: Path to video file
        threshold: Scene detection sensitivity (15-40, default 27)
        min_duration: Minimum clip duration in seconds
        footage_type: 'bw' or 'color' - adjusts detection parameters
        adaptive_threshold: Auto-adjust threshold based on footage type
    
    Returns:
        List of timestamp dicts with start, end, duration
    """
    # Adjust threshold based on footage type if adaptive
    if adaptive_threshold:
        if footage_type == 'bw':
            # B&W footage often needs higher threshold (subtler changes)
            threshold = threshold * 1.2
            print(f"  Adjusted threshold for B&W: {threshold:.1f}")
        elif footage_type == 'color':
            # Color footage works well with standard threshold
            pass
    
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    
    # CRITICAL: Use luma_only=True for B&W footage (equivalent to -l flag)
    if footage_type == 'bw':
        # B&W footage - only analyze luminance channel (ignore color noise)
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                luma_only=True,          # Analyze only brightness channel
                min_scene_len=15         # Longer minimum for B&W (avoid flicker)
            )
        )
        print(f"  Using luma-only detection for B&W")
    else:
        # Color footage - analyze all channels
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                luma_only=False          # Analyze R, G, B channels
            )
        )
    
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    
    timestamps = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec
        
        if duration >= min_duration:
            timestamps.append({
                'clip_num': i + 1,
                'start': start_sec,
                'end': end_sec,
                'duration': duration,
                'footage_type': footage_type
            })
    
    return timestamps


def cut_video_with_timestamps(video_path, timestamps, output_dir, stem_prefix=None):
    """
    Apply timestamps to cut video using stream copy (fast, lossless).
    
    Args:
        video_path: Path to video file to cut
        timestamps: List of timestamp dicts from detect_scenes_get_timestamps
        output_dir: Directory to save clips
        stem_prefix: Optional prefix for output filenames
    
    Returns:
        List of paths to created clips
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = Path(video_path)
    stem = stem_prefix if stem_prefix else video_path.stem
    clips = []
    
    for ts in timestamps:
        # Include footage type in filename if present
        footage_suffix = f"_{ts.get('footage_type', 'color')}" if 'footage_type' in ts else ""
        output_file = output_dir / f"{stem}_clip_{ts['clip_num']:03d}{footage_suffix}.mp4"
        
        cmd = [
            'ffmpeg',
            '-ss', str(ts['start']),
            '-i', str(video_path),
            '-t', str(ts['duration']),
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            '-y',  # Overwrite without asking
            str(output_file)
        ]
        
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        if result.returncode == 0:
            clips.append(output_file)
        else:
            print(f"Warning: Failed to create {output_file.name}", file=sys.stderr)
    
    return clips


def resize_video(input_path, 
                output_path, 
                height=480, 
                crf=20, 
                preset='medium',
                footage_type='color',
                enhance_bw=False):
    """
    Resize video to lower resolution.
    
    Args:
        input_path: Source video path
        output_path: Destination video path
        height: Target height in pixels
        crf: Quality (18=high, 23=good, 28=lower)
        preset: Encoding speed (ultrafast, fast, medium, slow)
        footage_type: 'bw' or 'color'
        enhance_bw: Apply contrast enhancement for B&W footage
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build video filter
    vf_filters = [f'scale=-1:{height}']
    
    # Optional: Enhance B&W footage for better scene detection
    if footage_type == 'bw' and enhance_bw:
        # Increase contrast and normalize levels for B&W
        vf_filters.append('eq=contrast=1.2:brightness=0.05')
        vf_filters.append('normalize')
        print(f"  Applying B&W enhancement: contrast boost + normalize")
    
    vf_string = ','.join(vf_filters)
    
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', vf_string,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', preset,
        '-an',
        '-y',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to resize {input_path}")
    
    return output_path


def auto_detect_footage_type(video_path, sample_frames=10):
    """
    Attempt to auto-detect if footage is B&W or color.
    Samples frames and checks color saturation.
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample
    
    Returns:
        'bw' or 'color'
    """
    try:
        import tempfile
        import cv2
        import numpy as np
    except ImportError:
        print("Warning: opencv-python required for auto-detection, defaulting to color", file=sys.stderr)
        return 'color'
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Extract sample frames
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,30))',  # Every 30th frame
            '-frames:v', str(sample_frames),
            '-vsync', '0',
            f'{temp_dir}/sample_%03d.png'
        ]
        
        subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        # Analyze color saturation
        saturations = []
        for frame_file in sorted(temp_dir.glob('sample_*.png')):
            img = cv2.imread(str(frame_file))
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1].mean()
                saturations.append(saturation)
        
        # Cleanup
        for f in temp_dir.glob('*.png'):
            f.unlink()
        temp_dir.rmdir()
        
        if not saturations:
            return 'color'  # Default to color if detection fails
        
        avg_saturation = np.mean(saturations)
        
        # Threshold: B&W footage typically has saturation < 20-30
        if avg_saturation < 25:
            return 'bw'
        else:
            return 'color'
    
    except Exception as e:
        print(f"Auto-detection failed: {e}, defaulting to color", file=sys.stderr)
        return 'color'


def process_single_video(video_hires,
                        video_lowres=None,
                        output_dir_hires=None,
                        output_dir_lowres=None,
                        threshold=27,
                        min_duration=2.0,
                        lowres_height=480,
                        footage_type='color',
                        adaptive_threshold=False,
                        enhance_bw=False,
                        auto_detect=False):
    """
    Process single video: detect scenes on lowres, cut hires (and optionally lowres).
    
    Args:
        video_hires: Path to high-resolution video
        video_lowres: Path to low-res video (if None, will create from hires)
        output_dir_hires: Output directory for high-res clips
        output_dir_lowres: Output directory for low-res clips (None to skip)
        threshold: Scene detection threshold
        min_duration: Minimum clip duration
        lowres_height: Height for low-res video if creating
        footage_type: 'bw' or 'color'
        adaptive_threshold: Auto-adjust threshold based on footage type
        enhance_bw: Apply contrast enhancement for B&W footage
        auto_detect: Auto-detect if footage is B&W or color
    
    Returns:
        Dict with clips_hires, clips_lowres, timestamps
    """
    video_hires = Path(video_hires)
    
    # Auto-detect footage type if requested
    if auto_detect:
        detected_type = auto_detect_footage_type(video_hires)
        print(f"  Auto-detected footage type: {detected_type.upper()}")
        footage_type = detected_type
    
    # Create low-res version if not provided
    if video_lowres is None:
        temp_lowres = Path(f'/tmp/{video_hires.stem}_lowres.mp4')
        print(f"  Creating low-res version for scene detection ({footage_type.upper()})...")
        video_lowres = resize_video(
            video_hires, 
            temp_lowres, 
            height=lowres_height,
            footage_type=footage_type,
            enhance_bw=enhance_bw
        )
        cleanup_lowres = True
    else:
        video_lowres = Path(video_lowres)
        cleanup_lowres = False
    
    # Detect scenes on low-res
    print(f"  Detecting scenes on {video_lowres.name} ({footage_type.upper()})...")
    timestamps = detect_scenes_get_timestamps(
        video_lowres, 
        threshold, 
        min_duration,
        footage_type=footage_type,
        adaptive_threshold=adaptive_threshold
    )
    print(f"  Found {len(timestamps)} scenes")
    
    # Cut high-res video
    if output_dir_hires:
        print(f"  Cutting high-res clips...")
        clips_hires = cut_video_with_timestamps(
            video_hires, 
            timestamps, 
            output_dir_hires,
            stem_prefix=video_hires.stem
        )
        print(f"  Created {len(clips_hires)} high-res clips")
    else:
        clips_hires = []
    
    # Cut low-res video (optional)
    if output_dir_lowres:
        print(f"  Cutting low-res clips...")
        clips_lowres = cut_video_with_timestamps(
            video_lowres,
            timestamps,
            output_dir_lowres,
            stem_prefix=video_hires.stem
        )
        print(f"  Created {len(clips_lowres)} low-res clips")
    else:
        clips_lowres = []
    
    # Cleanup temp low-res file
    if cleanup_lowres and video_lowres.exists():
        video_lowres.unlink()
    
    return {
        'video_hires': str(video_hires),
        'footage_type': footage_type,
        'clips_hires': [str(c) for c in clips_hires],
        'clips_lowres': [str(c) for c in clips_lowres],
        'timestamps': timestamps
    }


def process_directory(hires_dir,
                     lowres_dir=None,
                     output_dir_hires=None,
                     output_dir_lowres=None,
                     threshold=27,
                     min_duration=2.0,
                     lowres_height=480,
                     create_lowres=False,
                     footage_type='color',
                     adaptive_threshold=False,
                     enhance_bw=False,
                     auto_detect=False,
                     footage_map=None):
    """
    Process directory of videos.
    
    Args:
        hires_dir: Directory with high-resolution videos
        lowres_dir: Directory with low-res videos (None to create from hires)
        output_dir_hires: Output directory for high-res clips
        output_dir_lowres: Output directory for low-res clips (None to skip)
        threshold: Scene detection threshold
        min_duration: Minimum clip duration
        lowres_height: Height for low-res videos if creating
        create_lowres: Whether to create and save low-res versions
        footage_type: 'bw', 'color', or 'mixed' (use footage_map)
        adaptive_threshold: Auto-adjust threshold based on footage type
        enhance_bw: Apply contrast enhancement for B&W footage
        auto_detect: Auto-detect if footage is B&W or color for each file
        footage_map: Dict mapping filenames to 'bw' or 'color' (for mixed)
    
    Returns:
        List of processing results
    """
    hires_dir = Path(hires_dir)
    
    # Find all video files
    video_files = list(hires_dir.glob('*.mp4')) + list(hires_dir.glob('*.mov'))
    video_files.sort()
    
    if not video_files:
        print(f"No video files found in {hires_dir}")
        return []
    
    print(f"Found {len(video_files)} videos to process")
    print("="*60)
    
    # Create lowres directory if needed
    if create_lowres and lowres_dir:
        lowres_dir = Path(lowres_dir)
        lowres_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating low-res versions in {lowres_dir}")
        print("="*60)
    
    results = []
    
    for i, video_hires in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing {video_hires.name}")
        print("-"*60)
        
        # Determine footage type for this video
        if footage_type == 'mixed' and footage_map:
            video_footage_type = footage_map.get(video_hires.name, 'color')
            print(f"  Footage type from map: {video_footage_type.upper()}")
        else:
            video_footage_type = footage_type
        
        # Determine lowres video path
        if lowres_dir:
            video_lowres = Path(lowres_dir) / video_hires.name
            if not video_lowres.exists() and create_lowres:
                print(f"  Creating {video_lowres.name}...")
                resize_video(
                    video_hires, 
                    video_lowres, 
                    height=lowres_height,
                    footage_type=video_footage_type,
                    enhance_bw=enhance_bw
                )
        else:
            video_lowres = None
        
        # Determine output directories for this video
        if output_dir_hires:
            out_hires = Path(output_dir_hires) / video_hires.stem
        else:
            out_hires = None
        
        if output_dir_lowres:
            out_lowres = Path(output_dir_lowres) / video_hires.stem
        else:
            out_lowres = None
        
        # Process video
        try:
            result = process_single_video(
                video_hires=video_hires,
                video_lowres=video_lowres,
                output_dir_hires=out_hires,
                output_dir_lowres=out_lowres,
                threshold=threshold,
                min_duration=min_duration,
                lowres_height=lowres_height,
                footage_type=video_footage_type,
                adaptive_threshold=adaptive_threshold,
                enhance_bw=enhance_bw,
                auto_detect=auto_detect
            )
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
    
    return results


def load_footage_map(map_file):
    """
    Load footage type mapping from JSON file.
    
    Format:
    {
        "video1.mp4": "bw",
        "video2.mp4": "color",
        "video3.mp4": "bw"
    }
    """
    with open(map_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Detect scenes on low-res video and apply cuts to high-res originals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single B&W video (uses luma-only detection)
  %(prog)s -i video_4k.mp4 -o clips_4k/ --bw
  
  # Process color video with custom threshold
  %(prog)s -i video_4k.mp4 -o clips_4k/ --color --threshold 30
  
  # Process directory with mixed B&W and color footage
  %(prog)s -d videos_4k/ -o clips_4k/ --mixed --footage-map map.json
  
  # Auto-detect footage type for each video
  %(prog)s -d videos_4k/ -o clips_4k/ --auto-detect
  
  # B&W with adaptive threshold and enhancement
  %(prog)s -d videos_4k/ -o clips_4k/ --bw --adaptive --enhance-bw
  
  # Create low-res versions for color footage
  %(prog)s -d videos_4k/ -ld videos_480p/ -o clips_4k/ --create-lowres --color

Footage Type Options:
  --bw           Black and white footage (uses luma-only detection, higher threshold)
  --color        Color footage (default, analyzes all color channels)
  --mixed        Mixed footage types (requires --footage-map)
  --auto-detect  Auto-detect B&W vs color for each video
  
B&W Detection Improvements:
  When using --bw, the following optimizations are applied:
  - luma_only=True: Only analyzes brightness channel (ignores color noise)
  - Higher threshold: Default 32 vs 27 for color
  - Longer min_scene_len: 15 frames to avoid flicker/grain cuts
  - Optional --enhance-bw: Boost contrast for better detection
  
Recommended Thresholds:
  B&W footage:   28-35 (less sensitive due to subtle changes)
  Color footage: 25-30 (standard sensitivity)
  
Footage Map JSON Format (for --mixed):
  {
    "section_001.mp4": "bw",
    "section_002.mp4": "color",
    "section_003.mp4": "bw"
  }
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input',
                            help='Single high-resolution video file')
    input_group.add_argument('-d', '--directory',
                            help='Directory containing high-resolution videos')
    
    # Low-res input options
    parser.add_argument('-l', '--lowres',
                       help='Low-res video file (for -i) or directory (for -d)')
    parser.add_argument('--create-lowres', action='store_true',
                       help='Create and save low-res versions (requires -ld)')
    parser.add_argument('-ld', '--lowres-dir',
                       help='Directory to save/find low-res versions')
    parser.add_argument('--lowres-height', type=int, default=480,
                       help='Height for low-res videos (default: 480)')
    
    # Output options
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for high-res clips')
    parser.add_argument('-ol', '--output-lowres',
                       help='Output directory for low-res clips (optional)')
    
    # Footage type options
    footage_group = parser.add_mutually_exclusive_group()
    footage_group.add_argument('--bw', action='store_true',
                              help='Black and white footage (luma-only detection)')
    footage_group.add_argument('--color', action='store_true',
                              help='Color footage (default)')
    footage_group.add_argument('--mixed', action='store_true',
                              help='Mixed B&W and color (requires --footage-map)')
    footage_group.add_argument('--auto-detect', action='store_true',
                              help='Auto-detect B&W vs color for each video')
    
    parser.add_argument('--footage-map',
                       help='JSON file mapping filenames to footage types (for --mixed)')
    
    # Detection parameters
    parser.add_argument('-t', '--threshold', type=float, default=27.0,
                       help='Scene detection threshold (15-40, default: 27)')
    parser.add_argument('--adaptive', action='store_true',
                       help='Auto-adjust threshold based on footage type')
    parser.add_argument('--enhance-bw', action='store_true',
                       help='Enhance contrast for B&W footage (helps detection)')
    parser.add_argument('-m', '--min-duration', type=float, default=2.0,
                       help='Minimum clip duration in seconds (default: 2.0)')
    
    # Output options
    parser.add_argument('--save-timestamps', action='store_true',
                       help='Save timestamps as JSON files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Detect scenes but don\'t cut videos')
    
    args = parser.parse_args()
    
    # Validation
    if args.create_lowres and not args.lowres_dir:
        parser.error('--create-lowres requires --lowres-dir')
    
    if args.mixed and not args.footage_map:
        parser.error('--mixed requires --footage-map')
    
    # Determine footage type
    if args.bw:
        footage_type = 'bw'
    elif args.mixed:
        footage_type = 'mixed'
    else:
        footage_type = 'color'  # Default
    
    # Load footage map if provided
    footage_map = None
    if args.footage_map:
        footage_map = load_footage_map(args.footage_map)
        print(f"Loaded footage map with {len(footage_map)} entries")
    
    # Adjust default threshold for B&W if not explicitly set
    threshold = args.threshold
    if args.bw and args.threshold == 27.0:
        threshold = 32.0  # Higher default for B&W
        print(f"Using default B&W threshold: {threshold}")
    
    # Single video mode
    if args.input:
        print("Processing single video")
        print("="*60)
        
        output_hires = None if args.dry_run else args.output
        output_lowres = None if args.dry_run else args.output_lowres
        
        result = process_single_video(
            video_hires=args.input,
            video_lowres=args.lowres,
            output_dir_hires=output_hires,
            output_dir_lowres=output_lowres,
            threshold=threshold,
            min_duration=args.min_duration,
            lowres_height=args.lowres_height,
            footage_type=footage_type,
            adaptive_threshold=args.adaptive,
            enhance_bw=args.enhance_bw,
            auto_detect=args.auto_detect
        )
        
        # Save timestamps if requested
        if args.save_timestamps:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            json_file = output_path / f"{Path(args.input).stem}_timestamps.json"
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nTimestamps saved to {json_file}")
        
        print("\n" + "="*60)
        print("COMPLETE")
        print(f"Footage type: {result['footage_type'].upper()}")
        print(f"High-res clips: {len(result['clips_hires'])}")
        if result['clips_lowres']:
            print(f"Low-res clips: {len(result['clips_lowres'])}")
    
    # Directory mode
    else:
        print("Processing directory")
        print("="*60)
        
        output_hires = None if args.dry_run else args.output
        output_lowres = None if args.dry_run else args.output_lowres
        
        results = process_directory(
            hires_dir=args.directory,
            lowres_dir=args.lowres or args.lowres_dir,
            output_dir_hires=output_hires,
            output_dir_lowres=output_lowres,
            threshold=threshold,
            min_duration=args.min_duration,
            lowres_height=args.lowres_height,
            create_lowres=args.create_lowres,
            footage_type=footage_type,
            adaptive_threshold=args.adaptive,
            enhance_bw=args.enhance_bw,
            auto_detect=args.auto_detect,
            footage_map=footage_map
        )
        
        # Save all timestamps if requested
        if args.save_timestamps:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            json_file = output_path / 'all_timestamps.json'
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nAll timestamps saved to {json_file}")
        
        # Summary by footage type
        print("\n" + "="*60)
        print("COMPLETE")
        print(f"Processed {len(results)} videos")
        
        # Count by footage type
        bw_count = sum(1 for r in results if r.get('footage_type') == 'bw')
        color_count = sum(1 for r in results if r.get('footage_type') == 'color')
        
        if bw_count > 0:
            print(f"B&W videos: {bw_count}")
        if color_count > 0:
            print(f"Color videos: {color_count}")
        
        total_hires = sum(len(r['clips_hires']) for r in results)
        print(f"Total high-res clips: {total_hires}")
        if results and results[0]['clips_lowres']:
            total_lowres = sum(len(r['clips_lowres']) for r in results)
            print(f"Total low-res clips: {total_lowres}")


if __name__ == '__main__':
    main()