import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pickle
import gc

def parse_args():
    desc = """Detect humans in images and find the most unique images from a dataset.

    This tool uses a two-pass detection system:
    1. YOLO11 for fast initial human detection
    2. Moondream2 VLM as fallback for edge cases (artistic/stylized humans)

    Use --keep to specify which images to keep:
    - "humans" (default): Keep images WITH humans, discard images without
    - "no_humans": Keep images WITHOUT humans, discard images with humans

    Additional filters:
    - --exclude_text: Remove images containing text (uses EasyOCR)

    After filtering, it finds the most unique images using embeddings (CLIP or DINOv2)
    with Farthest Point Sampling for diversity selection.
    """
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_folder', type=str,
        default='./input/',
        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('-o', '--output_folder', type=str,
        default='./output/',
        help='Directory path to the outputs folder. (default: %(default)s)')

    parser.add_argument('--verbose', action='store_true',
        help='Print progress to console.')

    parser.add_argument('--file_extension', type=str,
        default='png',
        help='Output file extension ["png","jpg"] (default: %(default)s)')

    # Mode selection
    parser.add_argument('--mode', type=str,
        default='full',
        choices=['full', 'human_filter', 'unique_only'],
        help='Processing mode: "full" (filter + unique), "human_filter" (filter only), "unique_only" (skip filtering). (default: %(default)s)')

    # Human detection options
    parser.add_argument('--human_detector', type=str,
        default='hybrid',
        choices=['yolo', 'moondream', 'hybrid'],
        help='Human detection method: "yolo" (fast), "moondream" (VLM), "hybrid" (YOLO + Moondream fallback). (default: %(default)s)')

    parser.add_argument('--yolo_model', type=str,
        default='yolo11n.pt',
        help='YOLO model to use. Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt (default: %(default)s)')

    parser.add_argument('--yolo_confidence', type=float,
        default=0.25,
        help='YOLO confidence threshold for person detection. (default: %(default)s)')

    parser.add_argument('--yolo_batch_size', type=int,
        default=8,
        help='Batch size for YOLO detection. Lower values use less memory. (default: %(default)s)')

    parser.add_argument('--moondream_model', type=str,
        default='moondream-2b-int8',
        help='Moondream model variant. Options: moondream-2b-int8, moondream-2b-fp16 (default: %(default)s)')

    # Text detection options
    parser.add_argument('--exclude_text', action='store_true',
        help='Exclude images containing text.')

    parser.add_argument('--text_detector', type=str,
        default='east',
        choices=['east', 'easyocr', 'paddleocr', 'moondream'],
        help='Text detection method: "east" (fast, OpenCV built-in), "easyocr", "paddleocr" (fast), or "moondream" (VLM). (default: %(default)s)')

    parser.add_argument('--text_confidence', type=float,
        default=0.5,
        help='Confidence threshold for text detection. (default: %(default)s)')

    parser.add_argument('--min_text_area', type=float,
        default=0.001,
        help='Minimum text bounding box area as fraction of image (0.0-1.0). Helps ignore tiny text. (default: %(default)s)')

    # Embedding options
    parser.add_argument('--embedder', type=str,
        default='clip',
        choices=['clip', 'dinov2'],
        help='Embedding model for uniqueness: "clip" or "dinov2". (default: %(default)s)')

    parser.add_argument('--num_unique', type=int,
        default=100,
        help='Number of unique images to select. (default: %(default)s)')

    parser.add_argument('--selection_method', type=str,
        default='fps',
        choices=['fps', 'kmedoids'],
        help='Uniqueness selection: "fps" (Farthest Point Sampling) or "kmedoids". (default: %(default)s)')

    # Performance options
    parser.add_argument('--device', type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device for inference. "auto" detects best available. (default: %(default)s)')

    parser.add_argument('--batch_size', type=int,
        default=32,
        help='Batch size for embedding extraction. (default: %(default)s)')

    # Caching options
    parser.add_argument('--cache_embeddings', action='store_true',
        help='Cache embeddings to disk for reuse.')

    parser.add_argument('--cache_dir', type=str,
        default='./.embedding_cache/',
        help='Directory for embedding cache. (default: %(default)s)')

    # Filtering direction
    parser.add_argument('--keep', type=str,
        default='humans',
        choices=['humans', 'no_humans'],
        help='Which images to keep: "humans" (keep images with humans) or "no_humans" (keep images without humans). (default: %(default)s)')

    parser.add_argument('--save_discarded', action='store_true',
        help='Also save discarded images to a separate folder.')

    args = parser.parse_args()
    return args


def get_device(device_arg):
    """Detect and return the best available device."""
    import torch

    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def clear_memory(verbose=False):
    """Clear memory and GPU cache between processing phases."""
    import torch

    if verbose:
        print("Clearing memory...")

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class YOLODetector:
    """YOLO-based human detector using ultralytics."""

    def __init__(self, model_name='yolo11n.pt', confidence=0.25, device='cpu', verbose=False):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device
        self.verbose = verbose
        # Person class is 0 in COCO
        self.person_class = 0

    def detect_human(self, image_path):
        """Returns True if a human is detected in the image."""
        results = self.model(image_path, conf=self.confidence, device=self.device, verbose=False)

        for result in results:
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy()
                if self.person_class in classes:
                    if self.verbose:
                        print(f"\t[YOLO] Human detected in {os.path.basename(image_path)}")
                    return True
        return False

    def detect_batch(self, image_paths, batch_size=16, show_progress=True):
        """Batch detection for multiple images. Returns dict of {path: has_human}."""
        results_dict = {}

        # Process in small batches to avoid MPS memory issues
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="YOLO detection", total=(len(image_paths) + batch_size - 1) // batch_size)

        for i in iterator:
            batch_paths = image_paths[i:i+batch_size]
            results = self.model(batch_paths, conf=self.confidence, device=self.device, verbose=False)

            for result, path in zip(results, batch_paths):
                has_human = False
                if result.boxes is not None:
                    classes = result.boxes.cls.cpu().numpy()
                    if self.person_class in classes:
                        has_human = True
                        if self.verbose:
                            print(f"\t[YOLO] Human detected in {os.path.basename(path)}")
                results_dict[path] = has_human

            # Clear MPS cache between batches
            if self.device == 'mps':
                import torch
                torch.mps.empty_cache()

        return results_dict


class MoondreamDetector:
    """Moondream2 VLM-based human detector."""

    def __init__(self, model_name='moondream-2b-int8', device='cpu', verbose=False):
        import moondream as md
        self.model = md.vl(model=model_name)
        self.device = device
        self.verbose = verbose
        self.prompt = "Is there a human, person, or human body visible in this image? Answer only 'yes' or 'no'."

    def detect_human(self, image_path):
        """Returns True if a human is detected in the image."""
        from PIL import Image

        image = Image.open(image_path).convert('RGB')
        answer = self.model.query(image, self.prompt)['answer'].strip().lower()

        has_human = answer.startswith('yes')
        if self.verbose and has_human:
            print(f"\t[Moondream] Human detected in {os.path.basename(image_path)}: {answer}")

        return has_human


class HybridDetector:
    """Two-pass detector: YOLO first, Moondream fallback for edge cases."""

    def __init__(self, yolo_model='yolo11n.pt', yolo_confidence=0.25,
                 moondream_model='moondream-2b-int8', device='cpu', verbose=False):
        self.yolo = YOLODetector(yolo_model, yolo_confidence, device, verbose)
        self.moondream = None  # Lazy load
        self.moondream_model = moondream_model
        self.device = device
        self.verbose = verbose
        self._moondream_loaded = False

    def _ensure_moondream(self):
        if not self._moondream_loaded:
            if self.verbose:
                print("Loading Moondream2 model for fallback detection...")
            self.moondream = MoondreamDetector(self.moondream_model, self.device, self.verbose)
            self._moondream_loaded = True

    def detect_human(self, image_path):
        """Two-pass detection: YOLO first, Moondream if YOLO says no human."""
        # First pass: YOLO
        if self.yolo.detect_human(image_path):
            return True

        # Second pass: Moondream for edge cases
        self._ensure_moondream()
        return self.moondream.detect_human(image_path)

    def detect_batch(self, image_paths, batch_size=16, show_progress=True):
        """Batch detection with two-pass approach."""
        results = {}

        # First pass: YOLO (fast)
        if self.verbose:
            print("Pass 1: YOLO detection...")
        yolo_results = self.yolo.detect_batch(image_paths, batch_size=batch_size, show_progress=show_progress)

        # Separate humans from non-humans
        humans = [p for p, has_human in yolo_results.items() if has_human]
        non_humans = [p for p, has_human in yolo_results.items() if not has_human]

        # Mark YOLO-detected humans
        for path in humans:
            results[path] = True

        # Second pass: Moondream on YOLO non-humans
        if non_humans:
            if self.verbose:
                print(f"Pass 2: Moondream fallback on {len(non_humans)} images...")
            self._ensure_moondream()

            iterator = tqdm(non_humans, desc="Moondream check") if show_progress else non_humans
            for path in iterator:
                results[path] = self.moondream.detect_human(path)

        return results


class EASTTextDetector:
    """OpenCV EAST text detector - fast, no additional dependencies."""

    def __init__(self, confidence=0.5, min_text_area=0.001, device='cpu', verbose=False):
        self.confidence = confidence
        self.min_text_area = min_text_area
        self.verbose = verbose
        self.model_path = None
        self.net = None

    def _ensure_model(self):
        """Download and load EAST model if not already loaded."""
        if self.net is not None:
            return

        import urllib.request

        # EAST model path
        model_dir = Path.home() / '.cache' / 'east_text_detection'
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_dir / 'frozen_east_text_detection.pb'

        # Download if not exists
        if not self.model_path.exists():
            if self.verbose:
                print("Downloading EAST text detection model...")
            url = "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"
            urllib.request.urlretrieve(url, self.model_path)

        if self.verbose:
            print("Loading EAST model...")
        self.net = cv2.dnn.readNet(str(self.model_path))

    def detect_text(self, image_path):
        """Returns True if text is detected in the image."""
        self._ensure_model()

        img = cv2.imread(image_path)
        if img is None:
            return False

        orig_h, orig_w = img.shape[:2]
        img_area = orig_h * orig_w

        # EAST requires dimensions to be multiples of 32
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32
        new_w = max(new_w, 32)
        new_h = max(new_h, 32)

        ratio_w = orig_w / float(new_w)
        ratio_h = orig_h / float(new_h)

        # Resize and create blob
        resized = cv2.resize(img, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h),
                                      (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Run detection
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

        # Decode predictions
        rows, cols = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(rows):
            scores_data = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(cols):
                if scores_data[x] < self.confidence:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate bounding box area
                bbox_area = (w * ratio_w) * (h * ratio_h)
                area_fraction = bbox_area / img_area

                if area_fraction >= self.min_text_area:
                    if self.verbose:
                        print(f"\t[EAST] Text detected in {os.path.basename(image_path)} (conf: {scores_data[x]:.2f}, area: {area_fraction:.4f})")
                    return True

        return False

    def detect_batch(self, image_paths, show_progress=True):
        """Batch detection for multiple images."""
        results = {}
        iterator = tqdm(image_paths, desc="EAST text detection") if show_progress else image_paths

        for path in iterator:
            results[path] = self.detect_text(path)

        return results


class PaddleOCRTextDetector:
    """PaddleOCR-based text detector - faster than EasyOCR."""

    def __init__(self, confidence=0.5, min_text_area=0.001, device='cpu', verbose=False):
        from paddleocr import PaddleOCR

        self.confidence = confidence
        self.min_text_area = min_text_area
        self.verbose = verbose

        # Initialize PaddleOCR (detection only mode for speed)
        use_gpu = device == 'cuda'
        if verbose:
            print(f"Loading PaddleOCR (GPU: {use_gpu})...")
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=use_gpu,
                             show_log=False, det_db_score_mode='slow')

    def detect_text(self, image_path):
        """Returns True if text is detected in the image."""
        img = cv2.imread(image_path)
        if img is None:
            return False

        img_area = img.shape[0] * img.shape[1]

        # Run detection
        result = self.ocr.ocr(image_path, cls=False, rec=False)

        if result is None or len(result) == 0 or result[0] is None:
            return False

        for detection in result[0]:
            bbox = detection
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            bbox_width = max(x_coords) - min(x_coords)
            bbox_height = max(y_coords) - min(y_coords)
            bbox_area = bbox_width * bbox_height
            area_fraction = bbox_area / img_area

            if area_fraction >= self.min_text_area:
                if self.verbose:
                    print(f"\t[PaddleOCR] Text detected in {os.path.basename(image_path)} (area: {area_fraction:.4f})")
                return True

        return False

    def detect_batch(self, image_paths, show_progress=True):
        """Batch detection for multiple images."""
        results = {}
        iterator = tqdm(image_paths, desc="PaddleOCR text detection") if show_progress else image_paths

        for path in iterator:
            results[path] = self.detect_text(path)

        return results


class EasyOCRTextDetector:
    """EasyOCR-based text detector."""

    def __init__(self, confidence=0.5, min_text_area=0.001, device='cpu', verbose=False):
        import easyocr

        self.confidence = confidence
        self.min_text_area = min_text_area
        self.verbose = verbose

        # Use GPU only for CUDA (MPS can be unstable with EasyOCR)
        gpu = device == 'cuda'
        if verbose:
            print(f"Loading EasyOCR (GPU: {gpu})...")
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)

    def detect_text(self, image_path):
        """Returns True if text is detected in the image."""
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return False

        img_area = img.shape[0] * img.shape[1]

        # Detect text
        results = self.reader.readtext(image_path)

        for (bbox, text, confidence) in results:
            if confidence < self.confidence:
                continue

            # Calculate bounding box area
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            bbox_width = max(x_coords) - min(x_coords)
            bbox_height = max(y_coords) - min(y_coords)
            bbox_area = bbox_width * bbox_height
            area_fraction = bbox_area / img_area

            if area_fraction >= self.min_text_area:
                if self.verbose:
                    print(f"\t[EasyOCR] Text detected in {os.path.basename(image_path)}: '{text}' (conf: {confidence:.2f}, area: {area_fraction:.4f})")
                return True

        return False

    def detect_batch(self, image_paths, show_progress=True):
        """Batch detection for multiple images."""
        results = {}
        iterator = tqdm(image_paths, desc="Text detection") if show_progress else image_paths

        for path in iterator:
            results[path] = self.detect_text(path)

        return results


class MoondreamTextDetector:
    """Moondream2 VLM-based text detector."""

    def __init__(self, model_name='moondream-2b-int8', device='cpu', verbose=False):
        import moondream as md

        self.model = md.vl(model=model_name)
        self.device = device
        self.verbose = verbose
        self.prompt = "Is there any text, words, letters, or numbers visible in this image? Answer only 'yes' or 'no'."

    def detect_text(self, image_path):
        """Returns True if text is detected in the image."""
        from PIL import Image

        image = Image.open(image_path).convert('RGB')
        answer = self.model.query(image, self.prompt)['answer'].strip().lower()

        has_text = answer.startswith('yes')
        if self.verbose and has_text:
            print(f"\t[Moondream] Text detected in {os.path.basename(image_path)}: {answer}")

        return has_text

    def detect_batch(self, image_paths, show_progress=True):
        """Batch detection for multiple images."""
        results = {}
        iterator = tqdm(image_paths, desc="Text detection (Moondream)") if show_progress else image_paths

        for path in iterator:
            results[path] = self.detect_text(path)

        return results


class CLIPEmbedder:
    """CLIP-based image embedder."""

    def __init__(self, device='cpu', verbose=False):
        import torch
        from transformers import CLIPProcessor, CLIPModel

        self.device = device
        self.verbose = verbose

        if verbose:
            print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(device)
        self.model.eval()

    def embed(self, image_paths, batch_size=32, show_progress=True):
        """Extract embeddings for a list of images."""
        import torch
        from PIL import Image

        embeddings = []

        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting CLIP embeddings")

        with torch.no_grad():
            for i in iterator:
                batch_paths = image_paths[i:i+batch_size]
                images = [Image.open(p).convert('RGB') for p in batch_paths]

                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model.get_image_features(**inputs)
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)

                embeddings.append(outputs.cpu().numpy())

        return np.vstack(embeddings)


class DINOv2Embedder:
    """DINOv2-based image embedder."""

    def __init__(self, device='cpu', verbose=False):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.verbose = verbose

        if verbose:
            print("Loading DINOv2 model...")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large")
        self.model.to(device)
        self.model.eval()

    def embed(self, image_paths, batch_size=32, show_progress=True):
        """Extract embeddings for a list of images."""
        import torch
        from PIL import Image

        embeddings = []

        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting DINOv2 embeddings")

        with torch.no_grad():
            for i in iterator:
                batch_paths = image_paths[i:i+batch_size]
                images = [Image.open(p).convert('RGB') for p in batch_paths]

                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                # Use CLS token
                cls_embeddings = outputs.last_hidden_state[:, 0]
                cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)

                embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(embeddings)


class EmbeddingCache:
    """Cache for storing and loading embeddings."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.pkl"
        self.metadata_file = self.cache_dir / "metadata.json"

    def save(self, embeddings, image_paths, embedder_name):
        """Save embeddings to cache."""
        data = {
            'embeddings': embeddings,
            'image_paths': image_paths,
            'embedder': embedder_name
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)

        metadata = {
            'num_images': len(image_paths),
            'embedder': embedder_name,
            'embedding_dim': embeddings.shape[1]
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, embedder_name):
        """Load embeddings from cache if valid."""
        if not self.cache_file.exists():
            return None, None

        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)

        if data['embedder'] != embedder_name:
            return None, None

        return data['embeddings'], data['image_paths']


def farthest_point_sampling(embeddings, num_samples, seed=None):
    """
    Select the most diverse subset using Farthest Point Sampling.

    Args:
        embeddings: (N, D) numpy array of embeddings
        num_samples: number of samples to select
        seed: random seed for reproducibility

    Returns:
        indices of selected samples
    """
    n = len(embeddings)
    if num_samples >= n:
        return list(range(n))

    # Start with random point
    rng = np.random.default_rng(seed)
    selected = [rng.integers(n)]

    # Distance to nearest selected point for each point
    distances = np.full(n, np.inf)

    for _ in tqdm(range(num_samples - 1), desc="Farthest Point Sampling"):
        # Update distances with new selected point
        last_selected = selected[-1]
        new_distances = np.linalg.norm(embeddings - embeddings[last_selected], axis=1)
        distances = np.minimum(distances, new_distances)

        # Select point farthest from all selected points
        distances[selected] = -1  # Exclude already selected
        next_idx = np.argmax(distances)
        selected.append(next_idx)

    return selected


def kmedoids_sampling(embeddings, num_samples, seed=None):
    """
    Select diverse subset using K-Medoids clustering.

    Args:
        embeddings: (N, D) numpy array of embeddings
        num_samples: number of samples (clusters) to select
        seed: random seed for reproducibility

    Returns:
        indices of medoid samples
    """
    from sklearn_extra.cluster import KMedoids

    n = len(embeddings)
    if num_samples >= n:
        return list(range(n))

    kmedoids = KMedoids(n_clusters=num_samples, random_state=seed, metric='cosine')
    kmedoids.fit(embeddings)

    return list(kmedoids.medoid_indices_)


def save_image(img, path, filename, file_extension):
    """Save image with specified format."""
    if file_extension == "png":
        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        new_file = os.path.splitext(filename)[0] + ".jpg"
        cv2.imwrite(os.path.join(path, new_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def collect_image_paths(input_folder, verbose=False):
    """Collect all image paths from input folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = []

    for root, subdirs, files in os.walk(input_folder):
        if verbose:
            print(f'Scanning: {root}')

        for filename in files:
            if filename.startswith('.'):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                file_path = os.path.join(root, filename)
                image_paths.append(file_path)

    return image_paths


def main():
    args = parse_args()

    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Not a valid input folder: {args.input_folder}")
        return

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Detect device
    device = get_device(args.device)
    if args.verbose:
        print(f"Using device: {device}")

    # Collect image paths
    print("Collecting image paths...")
    image_paths = collect_image_paths(args.input_folder, args.verbose)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    # Phase 1: Human Detection
    non_human_paths = image_paths
    human_paths = []

    if args.mode in ['full', 'human_filter']:
        print("\n=== Phase 1: Human Detection ===")

        # Initialize detector
        if args.human_detector == 'yolo':
            detector = YOLODetector(args.yolo_model, args.yolo_confidence, device, args.verbose)
            results = detector.detect_batch(image_paths, batch_size=args.yolo_batch_size)
        elif args.human_detector == 'moondream':
            detector = MoondreamDetector(args.moondream_model, device, args.verbose)
            results = {}
            for path in tqdm(image_paths, desc="Moondream detection"):
                results[path] = detector.detect_human(path)
        else:  # hybrid
            detector = HybridDetector(
                args.yolo_model, args.yolo_confidence,
                args.moondream_model, device, args.verbose
            )
            results = detector.detect_batch(image_paths, batch_size=args.yolo_batch_size)

        # Separate results
        human_paths = [p for p, has_human in results.items() if has_human]
        non_human_paths = [p for p, has_human in results.items() if not has_human]

        print(f"\nDetection results:")
        print(f"  Images with humans: {len(human_paths)}")
        print(f"  Images without humans: {len(non_human_paths)}")

        # Determine which images to keep based on --keep flag
        if args.keep == 'humans':
            kept_paths = human_paths
            discarded_paths = non_human_paths
            kept_label = 'humans'
            discarded_label = 'no_humans'
        else:  # keep == 'no_humans'
            kept_paths = non_human_paths
            discarded_paths = human_paths
            kept_label = 'no_humans'
            discarded_label = 'humans'

        print(f"\nKeeping: {len(kept_paths)} images ({kept_label})")
        print(f"Discarding: {len(discarded_paths)} images ({discarded_label})")

        # Free memory from human detection models
        del detector
        clear_memory(args.verbose)

    # Phase 1.5: Text Detection (optional)
    text_paths = []
    if args.exclude_text and args.mode in ['full', 'human_filter']:
        print("\n=== Phase 1.5: Text Detection ===")

        # Use kept_paths if we did human filtering, otherwise all images
        paths_to_check = kept_paths if args.mode in ['full', 'human_filter'] else image_paths

        if args.text_detector == 'east':
            text_detector = EASTTextDetector(
                args.text_confidence, args.min_text_area, device, args.verbose
            )
        elif args.text_detector == 'paddleocr':
            text_detector = PaddleOCRTextDetector(
                args.text_confidence, args.min_text_area, device, args.verbose
            )
        elif args.text_detector == 'easyocr':
            text_detector = EasyOCRTextDetector(
                args.text_confidence, args.min_text_area, device, args.verbose
            )
        else:  # moondream
            text_detector = MoondreamTextDetector(args.moondream_model, device, args.verbose)

        text_results = text_detector.detect_batch(paths_to_check)

        # Separate images with and without text
        text_paths = [p for p, has_text in text_results.items() if has_text]
        no_text_paths = [p for p, has_text in text_results.items() if not has_text]

        print(f"\nText detection results:")
        print(f"  Images with text: {len(text_paths)}")
        print(f"  Images without text: {len(no_text_paths)}")

        # Update kept_paths to exclude images with text
        kept_paths = no_text_paths
        discarded_paths = discarded_paths + text_paths

        print(f"\nAfter text filtering: {len(kept_paths)} images remaining")

        # Free memory from text detection models
        del text_detector
        clear_memory(args.verbose)

    # Save discarded images if requested
    if args.mode in ['full', 'human_filter'] and args.save_discarded and discarded_paths:
        discarded_output = os.path.join(args.output_folder, 'discarded')
        os.makedirs(discarded_output, exist_ok=True)
        print(f"\nSaving discarded images to: {discarded_output}")
        for path in tqdm(discarded_paths, desc="Saving discarded images"):
            img = cv2.imread(path)
            if img is not None:
                save_image(img, discarded_output, os.path.basename(path), args.file_extension)

    # If human_filter only mode, save kept images and exit
    if args.mode == 'human_filter':
        kept_output = os.path.join(args.output_folder, kept_label)
        os.makedirs(kept_output, exist_ok=True)
        print(f"\nSaving kept images to: {kept_output}")
        for path in tqdm(kept_paths, desc="Saving kept images"):
            img = cv2.imread(path)
            if img is not None:
                save_image(img, kept_output, os.path.basename(path), args.file_extension)
        print("\nFiltering complete!")
        return

    # Phase 2: Uniqueness Analysis
    if args.mode in ['full', 'unique_only']:
        print("\n=== Phase 2: Uniqueness Analysis ===")

        # In unique_only mode, use all images; otherwise use kept_paths from Phase 1
        if args.mode == 'unique_only':
            candidate_paths = image_paths
            kept_label = 'all'
        else:
            candidate_paths = kept_paths

        if len(candidate_paths) == 0:
            print("No images to process for uniqueness. Exiting.")
            return

        if len(candidate_paths) <= args.num_unique:
            print(f"Only {len(candidate_paths)} images available, selecting all.")
            selected_paths = candidate_paths
        else:
            # Check cache
            embeddings = None
            if args.cache_embeddings:
                cache = EmbeddingCache(args.cache_dir)
                embeddings, cached_paths = cache.load(args.embedder)
                if embeddings is not None and set(cached_paths) == set(candidate_paths):
                    print("Using cached embeddings")
                else:
                    embeddings = None

            # Extract embeddings if not cached
            if embeddings is None:
                if args.embedder == 'clip':
                    embedder = CLIPEmbedder(device, args.verbose)
                else:
                    embedder = DINOv2Embedder(device, args.verbose)

                embeddings = embedder.embed(candidate_paths, args.batch_size)

                # Cache if requested
                if args.cache_embeddings:
                    cache = EmbeddingCache(args.cache_dir)
                    cache.save(embeddings, candidate_paths, args.embedder)
                    print(f"Embeddings cached to {args.cache_dir}")

            # Select unique images
            print(f"\nSelecting {args.num_unique} most unique images...")
            if args.selection_method == 'fps':
                selected_indices = farthest_point_sampling(embeddings, args.num_unique)
            else:
                selected_indices = kmedoids_sampling(embeddings, args.num_unique)

            selected_paths = [candidate_paths[i] for i in selected_indices]

        # Save unique images
        unique_output = os.path.join(args.output_folder, 'unique')
        os.makedirs(unique_output, exist_ok=True)
        print(f"\nSaving {len(selected_paths)} unique images to: {unique_output}")

        for path in tqdm(selected_paths, desc="Saving unique images"):
            img = cv2.imread(path)
            if img is not None:
                save_image(img, unique_output, os.path.basename(path), args.file_extension)

        # Save selection manifest
        manifest_path = os.path.join(unique_output, 'selection_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump({
                'total_images': len(image_paths),
                'kept_after_filter': len(candidate_paths),
                'filter_kept': kept_label if args.mode == 'unique_only' else args.keep,
                'selected_unique': len(selected_paths),
                'embedder': args.embedder,
                'selection_method': args.selection_method,
                'selected_files': [os.path.basename(p) for p in selected_paths]
            }, f, indent=2)

        print(f"\nSelection manifest saved to: {manifest_path}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
