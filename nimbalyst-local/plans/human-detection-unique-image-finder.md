---
planStatus:
  planId: plan-human-detection-unique-image-finder
  title: Human Detection & Unique Image Finder
  status: in-development
  planType: feature
  priority: high
  owner: derrickschultz
  tags:
    - image-processing
    - human-detection
    - text-detection
    - embeddings
    - machine-learning
    - video-generation
  created: "2025-12-26"
  updated: "2025-12-27T21:30:00.000Z"
  progress: 95
---

# Human Detection & Unique Image Finder

## Goals
- Filter large image datasets (thousands of images) to identify images containing humans
- Optionally exclude images containing text
- Find the most unique/diverse images from the filtered set using embeddings
- Provide curated starting frames for generative video models
- Support both GPU and CPU processing (including Apple Silicon M1/M2/M3)

## Overview

This tool addresses a multi-stage pipeline for curating image datasets:

1. **Human Detection Stage**: Identify images containing humans using YOLO + optional Moondream2 VLM fallback
2. **Text Detection Stage** (optional): Filter out images containing text using EAST, PaddleOCR, or EasyOCR
3. **Uniqueness Analysis Stage**: Use embedding-based similarity analysis to identify the most unique/diverse subset

## Decision: YOLO + Moondream2 Hybrid Approach

**Chosen approach**: Two-pass detection system

1. **First pass - YOLO11**: Fast detection catches obvious humans (~100 img/sec on GPU)
2. **Second pass - Moondream2**: VLM reviews images where YOLO found no humans, catches edge cases

**Rationale**:
- YOLO handles the bulk of detection quickly
- Moondream2 is lightweight (1.8B params, ~3-4GB VRAM) and catches artistic/stylized humans that YOLO misses
- Better accuracy than YOLO alone, much faster than VLM-only approach
- Both work on CPU with reasonable performance

### VLM Options Evaluated

| Model | Speed | Accuracy | Edge Cases | VRAM | Ease of Use |
|-------|-------|----------|------------|------|-------------|
| **Moondream2** | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★★★★ |
| **Florence-2** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ | ★★★☆☆ |
| **Qwen2-VL-2B** | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| **Qwen2-VL-7B** | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★☆☆☆ | ★★★★☆ |

## Decision: Text Detection Options

Added text detection to filter out images containing text (common in video frame extraction).

| Detector | Speed | Dependencies | Notes |
|----------|-------|--------------|-------|
| **EAST** (default) | Fastest | None (OpenCV built-in) | Downloads 100MB model on first run |
| **PaddleOCR** | Fast | `paddlepaddle paddleocr` | Good accuracy |
| **EasyOCR** | Slow | `easyocr` | Most accurate, full OCR |
| **Moondream** | Medium | Already loaded | Uses VLM |

## Implementation Details

### Script Structure (Implemented)
```
human_filter_unique.py
├── Human Detection Module
│   ├── YOLODetector class (primary, fast, batched)
│   ├── MoondreamDetector class (VLM fallback)
│   └── HybridDetector class (YOLO + Moondream2)
├── Text Detection Module
│   ├── EASTTextDetector class (fast, OpenCV built-in)
│   ├── PaddleOCRTextDetector class (fast)
│   ├── EasyOCRTextDetector class (accurate but slow)
│   └── MoondreamTextDetector class (VLM)
├── Embedding Module
│   ├── CLIPEmbedder class
│   ├── DINOv2Embedder class
│   └── EmbeddingCache (save/load embeddings)
├── Uniqueness Module
│   ├── farthest_point_sampling()
│   └── kmedoids_sampling()
├── Memory Management
│   ├── clear_memory() - clears GPU/MPS cache between phases
│   └── Batch processing with configurable batch sizes
└── CLI Interface
    └── argparse with standard conventions
```

### Command Line Interface (Implemented)
```bash
# Full pipeline: detect humans, exclude text, find unique
python human_filter_unique.py \
  --input_folder ./input/ \
  --output_folder ./output/ \
  --human_detector yolo \
  --exclude_text \
  --text_detector east \
  --num_unique 100 \
  --verbose

# Keep images WITH humans (default), exclude text
python human_filter_unique.py \
  --input_folder ./input/ \
  --output_folder ./output/ \
  --keep humans \
  --exclude_text \
  --num_unique 100

# Keep images WITHOUT humans
python human_filter_unique.py \
  --input_folder ./input/ \
  --output_folder ./output/ \
  --keep no_humans \
  --num_unique 100

# Human + text filtering only (no uniqueness)
python human_filter_unique.py \
  --input_folder ./input/ \
  --output_folder ./output/ \
  --mode human_filter \
  --exclude_text

# For M1 Macs with limited memory
python human_filter_unique.py \
  --input_folder ./input/ \
  --output_folder ./output/ \
  --human_detector yolo \
  --yolo_batch_size 4 \
  --exclude_text \
  --text_detector east \
  --batch_size 8 \
  --verbose
```

### Key CLI Options
| Option | Default | Description |
|--------|---------|-------------|
| `--keep` | `humans` | Which images to keep: `humans` or `no_humans` |
| `--human_detector` | `hybrid` | `yolo`, `moondream`, or `hybrid` |
| `--yolo_batch_size` | `8` | Batch size for YOLO (lower = less memory) |
| `--exclude_text` | `false` | Enable text detection filtering |
| `--text_detector` | `east` | `east`, `paddleocr`, `easyocr`, or `moondream` |
| `--embedder` | `clip` | `clip` or `dinov2` |
| `--num_unique` | `100` | Number of unique images to select |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |

### Dependencies
```
ultralytics>=8.0.0        # YOLO11
moondream                 # Moondream2 VLM (optional, for hybrid mode)
transformers>=4.30.0      # CLIP, DINOv2
torch>=2.0.0              # PyTorch
scikit-learn              # Clustering (already present)
tqdm                      # Progress bars

# Optional for text detection:
# paddlepaddle paddleocr  # For PaddleOCR
# easyocr                 # For EasyOCR
```

### M1/MPS Memory Management (Implemented)
- YOLO processes in configurable batches (default 8) instead of all at once
- MPS cache cleared between batches via `torch.mps.empty_cache()`
- Models deleted and garbage collected between phases
- EasyOCR forced to CPU on MPS (stability issues)

### Performance on M1 16GB
| Component | Batch Size | Speed | Memory |
|-----------|------------|-------|--------|
| YOLO Detection | 4-8 | ~20-30 img/sec | ~2GB |
| EAST Text Detection | 1 | ~15-20 img/sec | ~500MB |
| CLIP Embeddings | 8-16 | ~5-10 img/sec | ~2GB |

## Acceptance Criteria
- [x] Script follows codebase conventions (`--input_folder`, `--output_folder`, `--verbose`)
- [x] Human detection works with YOLO11
- [x] Optional Moondream2 VLM fallback for edge cases
- [x] Text detection with multiple backend options (EAST, PaddleOCR, EasyOCR, Moondream)
- [x] `--keep` flag to choose humans vs no_humans
- [x] Embedding extraction with CLIP or DINOv2 (user choice)
- [x] Farthest Point Sampling for uniqueness selection
- [x] K-Medoids as alternative selection method
- [x] Embedding cache to avoid reprocessing
- [x] Works on CPU (slower but functional)
- [x] Works on MPS (Apple Silicon) with memory management
- [x] Progress output with `--verbose`
- [x] Configurable batch sizes for memory-constrained systems
- [ ] Test with 10,000+ images
- [ ] Auto-generated documentation via existing workflow

## Known Issues
1. **MPS memory fragmentation**: On M1 Macs, memory may not be released cleanly. Use smaller batch sizes if crashes occur.
2. **EasyOCR slow**: Use EAST (default) or PaddleOCR for faster text detection.
3. **EAST model download**: First run downloads ~100MB model to `~/.cache/east_text_detection/`

## Future Enhancements
- Add GroundingDINO for more robust human detection
- Add FAISS indexing for very large datasets (100k+ images)
- Interactive review mode for borderline cases
- Export similarity visualization (t-SNE/UMAP plot)
- Integration with existing `dedupe.py` for a unified pipeline
- Staged processing mode for very large datasets
