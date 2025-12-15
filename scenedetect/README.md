# 8mm Scene Detection and Clip Extraction

Detect scenes on low-resolution video and apply precise cuts to high-resolution originals. Optimized for both black & white and color 8mm film footage with automatic detection and adaptive processing.

## Features

- **Dual-resolution workflow**: Detect scenes on 480p, cut 4K originals
- **B&W optimization**: Luma-only detection, higher thresholds, longer minimum scene length
- **Color support**: Full RGB analysis for color footage
- **Auto-detection**: Automatically determine if footage is B&W or color
- **Mixed archives**: Process collections with both B&W and color reels
- **Adaptive thresholds**: Automatically adjust sensitivity based on footage type
- **Contrast enhancement**: Optional boost for low-contrast B&W footage
- **Fast, lossless cutting**: Uses FFmpeg stream copy (no re-encoding)
- **Archival quality**: Keep 4K originals while processing at lower resolution

## Installation

```bash
# Install dependencies
pip install scenedetect[opencv] opencv-python numpy

# Make script executable (optional)
chmod +x scene_cut.py
```

## Quick Start

### Basic Examples

```bash
# Single B&W video
python scene_cut.py -i video_4k.mp4 -o clips_4k/ --bw

# Single color video
python scene_cut.py -i video_4k.mp4 -o clips_4k/ --color

# Directory with auto-detection
python scene_cut.py -d videos_4k/ -o clips_4k/ --auto-detect
```

## Usage

### Single Video Processing

#### B&W Footage
```bash
# Basic B&W processing (uses luma-only detection)
python scene_cut.py -i bw_video_4k.mp4 -o clips_4k/ --bw

# B&W with adaptive threshold
python scene_cut.py -i bw_video_4k.mp4 -o clips_4k/ --bw --adaptive

# B&W with contrast enhancement (for low-contrast footage)
python scene_cut.py -i bw_video_4k.mp4 -o clips_4k/ --bw --enhance-bw --adaptive

# B&W with custom threshold
python scene_cut.py -i bw_video_4k.mp4 -o clips_4k/ --bw --threshold 35
```

#### Color Footage
```bash
# Basic color processing
python scene_cut.py -i color_video_4k.mp4 -o clips_4k/ --color

# Color with custom threshold (more sensitive)
python scene_cut.py -i color_video_4k.mp4 -o clips_4k/ --color --threshold 25

# Color with adaptive threshold
python scene_cut.py -i color_video_4k.mp4 -o clips_4k/ --color --adaptive
```

#### With Pre-existing Low-Res Version
```bash
# Use existing 480p version for detection
python scene_cut.py -i video_4k.mp4 -l video_480p.mp4 -o clips_4k/ --bw
```

### Directory Processing

#### B&W Archive
```bash
# Process directory of B&W videos
python scene_cut.py -d bw_videos_4k/ -o clips_4k/ --bw --adaptive

# B&W with enhancement for grainy/low-contrast footage
python scene_cut.py -d bw_videos_4k/ -o clips_4k/ \
    --bw \
    --adaptive \
    --enhance-bw \
    --threshold 30
```

#### Color Archive
```bash
# Process directory of color videos
python scene_cut.py -d color_videos_4k/ -o clips_4k/ --color --threshold 27
```

#### Mixed Archive (B&W + Color)

**Option 1: Auto-detect (Recommended)**
```bash
# Automatically detect B&W vs color for each video
python scene_cut.py -d videos_4k/ -o clips_4k/ \
    --auto-detect \
    --adaptive \
    --enhance-bw
```

**Option 2: Use footage map**

Create `footage_map.json`:
```json
{
  "section_001.mp4": "bw",
  "section_002.mp4": "bw",
  "section_003.mp4": "color",
  "section_004.mp4": "bw",
  "section_005.mp4": "color"
}
```

Then run:
```bash
python scene_cut.py -d videos_4k/ -o clips_4k/ \
    --mixed \
    --footage-map footage_map.json \
    --adaptive
```

### Create Low-Res Versions for Future Use

```bash
# Create 480p versions and extract clips from both resolutions
python scene_cut.py -d videos_4k/ \
    -ld videos_480p/ \
    -o clips_4k/ \
    -ol clips_480p/ \
    --create-lowres \
    --bw \
    --adaptive

# Later, re-cut with different threshold using existing 480p
python scene_cut.py -d videos_4k/ \
    -ld videos_480p/ \
    -o clips_4k_v2/ \
    --bw \
    --threshold 35
```

### Testing and Dry Runs

```bash
# Test detection without cutting videos
python scene_cut.py -d videos_4k/ -o clips_4k/ \
    --bw \
    --threshold 30 \
    --dry-run \
    --save-timestamps

# This creates timestamps.json with scene information
# Review timestamps before committing to cuts
```

## Command-Line Options

### Input Options
```
-i, --input           Single high-resolution video file
-d, --directory       Directory containing high-resolution videos
-l, --lowres          Low-res video file (for -i) or directory (for -d)
-ld, --lowres-dir     Directory to save/find low-res versions
--create-lowres       Create and save low-res versions (requires -ld)
--lowres-height       Height for low-res videos (default: 480)
```

### Output Options
```
-o, --output          Output directory for high-res clips (required)
-ol, --output-lowres  Output directory for low-res clips (optional)
--save-timestamps     Save timestamps as JSON files
--dry-run            Detect scenes but don't cut videos
```

### Footage Type Options
```
--bw                 Black and white footage (luma-only detection)
--color              Color footage (default, analyzes all channels)
--mixed              Mixed B&W and color (requires --footage-map)
--auto-detect        Auto-detect B&W vs color for each video
--footage-map        JSON file mapping filenames to footage types
```

### Detection Parameters
```
-t, --threshold      Scene detection threshold (15-40, default: 27)
--adaptive          Auto-adjust threshold based on footage type
--enhance-bw        Enhance contrast for B&W footage
-m, --min-duration  Minimum clip duration in seconds (default: 2.0)
```

## Understanding Footage Types

### Black & White (`--bw`)

When processing B&W footage, the following optimizations are automatically applied:

1. **Luma-only detection** (`luma_only=True`)
   - Only analyzes brightness (Y) channel
   - Ignores color (U, V) channels
   - Prevents false cuts from color noise/artifacts

2. **Higher threshold** (default: 32 vs 27 for color)
   - B&W has more subtle scene changes
   - Higher threshold prevents cuts on grain/flicker

3. **Longer minimum scene length** (15 frames)
   - Avoids false cuts on film grain
   - Reduces flicker-induced false positives

4. **Adaptive threshold** (`--adaptive`)
   - Multiplies threshold by 1.2× for B&W
   - Example: 32 → 38.4

5. **Optional enhancement** (`--enhance-bw`)
   - Boosts contrast by 20%
   - Normalizes brightness histogram
   - Helps detection on low-contrast footage

### Color (`--color`)

Standard scene detection for color footage:
- Analyzes all RGB channels
- Lower threshold (default: 27)
- Detects both color and brightness changes

### Auto-Detection (`--auto-detect`)

Automatically determines footage type by:
1. Sampling 10 frames from each video
2. Measuring average color saturation
3. Classifying as B&W if saturation < 25
4. Applying appropriate detection settings

Requires `opencv-python` for analysis.

## Recommended Settings

### Clean B&W 8mm Footage
```bash
python scene_cut.py -d videos/ -o clips/ \
    --bw \
    --threshold 32 \
    --adaptive
```

### Grainy/Low-Contrast B&W Footage
```bash
python scene_cut.py -d videos/ -o clips/ \
    --bw \
    --threshold 30 \
    --enhance-bw \
    --adaptive
```

### Color 8mm Footage
```bash
python scene_cut.py -d videos/ -o clips/ \
    --color \
    --threshold 27
```

### Mixed Archive (Best for Most Cases)
```bash
python scene_cut.py -d videos/ -o clips/ \
    --auto-detect \
    --adaptive \
    --enhance-bw
```

## Threshold Guidelines

### For B&W Footage
- **28-30**: Very sensitive, more cuts (clean footage)
- **32-35**: Balanced (recommended for most B&W 8mm)
- **35-40**: Less sensitive (very grainy or noisy footage)

### For Color Footage
- **23-25**: Very sensitive, more cuts
- **27-30**: Balanced (recommended for most color 8mm)
- **30-35**: Less sensitive (grainy footage)

### Finding the Right Threshold

```bash
# Test different thresholds with dry-run
for threshold in 25 30 35; do
    python scene_cut.py -i test_video.mp4 -o test_clips/ \
        --bw \
        --threshold $threshold \
        --dry-run \
        --save-timestamps
    # Review timestamps_*.json files
done
```

## Output Structure

```
output_directory/
├── video_001/
│   ├── video_001_clip_001_bw.mp4
│   ├── video_001_clip_002_bw.mp4
│   └── video_001_clip_003_bw.mp4
├── video_002/
│   ├── video_002_clip_001_color.mp4
│   └── video_002_clip_002_color.mp4
└── all_timestamps.json  (if --save-timestamps used)
```

Clip filenames include:
- Source video stem
- Clip number (padded to 3 digits)
- Footage type suffix (`_bw` or `_color`)

## Typical Workflow

### 1. Initial Processing
```bash
# Process 4K archive with auto-detection
python scene_cut.py -d /Volumes/8mm/4k_sections/ \
    -o ./clips_4k/ \
    --auto-detect \
    --adaptive \
    --enhance-bw \
    --save-timestamps
```

### 2. Create 480p Archive for Fast Processing
```bash
# Create 480p versions for future use
python scene_cut.py -d /Volumes/8mm/4k_sections/ \
    -ld ./sections_480p/ \
    -o ./clips_4k/ \
    -ol ./clips_480p/ \
    --create-lowres \
    --auto-detect \
    --adaptive
```

### 3. Re-cut with Different Settings (Fast)
```bash
# Use existing 480p for detection, cut 4K with new threshold
python scene_cut.py -d /Volumes/8mm/4k_sections/ \
    -ld ./sections_480p/ \
    -o ./clips_4k_v2/ \
    --bw \
    --threshold 35
```

### 4. For CLIP Embeddings Pipeline
```bash
# Create both 4K (archival) and 480p (for embeddings)
python scene_cut.py -d /Volumes/8mm/4k_sections/ \
    -ld ./sections_480p/ \
    -o ./clips_4k/ \
    -ol ./clips_480p/ \
    --create-lowres \
    --auto-detect \
    --adaptive

# Use clips_480p for CLIP embeddings
# Keep clips_4k for final export
```

## Troubleshooting

### Too Many Short Clips (Over-segmentation)

**Problem**: Getting hundreds of 1-2 second clips from grain/flicker

**Solutions**:
```bash
# Increase threshold
python scene_cut.py -i video.mp4 -o clips/ --bw --threshold 35

# Increase minimum duration
python scene_cut.py -i video.mp4 -o clips/ --bw --min-duration 3.0

# Use both
python scene_cut.py -i video.mp4 -o clips/ --bw --threshold 35 --min-duration 3.0
```

### Missing Scene Changes (Under-segmentation)

**Problem**: Long clips that should be split

**Solutions**:
```bash
# Lower threshold
python scene_cut.py -i video.mp4 -o clips/ --bw --threshold 28

# For low-contrast B&W, use enhancement
python scene_cut.py -i video.mp4 -o clips/ --bw --threshold 30 --enhance-bw
```

### Auto-Detection Not Working

**Problem**: `opencv-python` not installed or detection failing

**Solution**:
```bash
# Install opencv
pip install opencv-python

# Or use manual footage type
python scene_cut.py -i video.mp4 -o clips/ --bw  # or --color
```

### FFmpeg Not Found

**Problem**: `ffmpeg` not in PATH

**Solution**:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Performance Notes

### Processing Time (100 clips)

| Stage | Resolution | Time |
|-------|-----------|------|
| Scene detection | 480p | ~5-10 min |
| Scene detection | 4K | ~15-25 min |
| Cutting (stream copy) | 4K | ~2-3 min |
| Total (480p detection) | - | ~7-13 min |
| Total (4K detection) | - | ~17-28 min |

**Recommendation**: Always detect on 480p for 3-4× speedup with identical results.

### Disk Space

For 100 three-minute 4K sections:
- 4K sections: ~50-60 GB
- 480p sections: ~15-20 GB
- 4K clips: ~50-60 GB (same as sections)
- 480p clips: ~15-20 GB

Total with both: ~130-160 GB

## Advanced Usage

### Custom FFmpeg Parameters

Edit the `cut_video_with_timestamps()` function to add custom FFmpeg options:

```python
cmd = [
    'ffmpeg',
    '-ss', str(ts['start']),
    '-i', str(video_path),
    '-t', str(ts['duration']),
    '-c:v', 'libx264',     # Re-encode instead of copy
    '-crf', '18',          # High quality
    '-preset', 'slow',     # Better compression
    str(output_file)
]
```

### Integration with Other Tools

```bash
# Generate embeddings from 480p clips
python scene_cut.py -d videos_4k/ -ld sections_480p/ -o clips_4k/ -ol clips_480p/ --create-lowres
python generate_embeddings.py -d clips_480p/ -o embeddings.npy

# Cluster by embeddings
python cluster_videos.py -e embeddings.npy -d clips_4k/
```

## Technical Details

### Scene Detection Algorithm

Uses PySceneDetect's `ContentDetector`:
- Compares consecutive frames using color/luma histograms
- Threshold determines how different frames must be
- B&W mode uses only luminance histogram (luma-only)
- Color mode uses RGB histograms

### Timestamp Transfer

1. Detect scenes on 480p video → get timestamps
2. Apply exact timestamps to 4K video using FFmpeg stream copy
3. Result: Fast detection, lossless 4K cuts

### B&W Enhancement Filter Chain

```
scale=-1:480,eq=contrast=1.2:brightness=0.05,normalize
```

- `scale`: Resize to 480p
- `eq`: Boost contrast 20%, lift shadows 5%
- `normalize`: Spread histogram to full range

## License

This script uses:
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (BSD 3-Clause)
- [FFmpeg](https://ffmpeg.org/) (LGPL/GPL)
- [OpenCV](https://opencv.org/) (Apache 2.0)

## Support

For issues or questions:
1. Check PySceneDetect docs: https://scenedetect.com/
2. Verify FFmpeg installation: `ffmpeg -version`
3. Test with dry-run to check detection quality

## Changelog

### v1.0.0
- Initial release
- B&W and color support
- Auto-detection
- Luma-only mode for B&W
- Adaptive thresholds
- Contrast enhancement
