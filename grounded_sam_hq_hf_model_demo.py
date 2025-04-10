import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="car. tire.")
parser.add_argument("--img-path", default="notebooks/images/truck.jpg")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hq_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hq_hiera_l.yaml")
parser.add_argument("--hq-token-only", action="store_true", help="Use only HQ output tokens for mask generation.")
parser.add_argument("--output-dir", default="outputs/test_grounded_sam2_hq_v2")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
TEXT_PROMPT = args.text_prompt
IMG_PATH = args.img_path
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
HQ_TOKEN_ONLY = args.hq_token_only
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
DUMP_JSON_RESULTS = not args.no_dump_json

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor (loading HQ V2 config/checkpoint)
print(f"Loading SAM2 model with HQ config ({SAM2_MODEL_CONFIG}) and checkpoint ({SAM2_CHECKPOINT})...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2-HQ model loaded.")

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)


# setup the input image and text prompt for SAM 2 HQ and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image_pil = Image.open(img_path).convert("RGB")
image_np = np.array(image_pil)

# Set image for SAM predictor
sam2_predictor.set_image(image_np)

# Process for GroundingDINO
inputs = processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image_pil.size[::-1]]
)

"""
Results is a list of dict with the following structure:
[
    {
        'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
        'labels': ['car', 'tire', 'tire', 'tire'], 
        'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                        [1392.4701,  554.4064, 1628.6133,  777.5872],
                        [ 436.1182,  621.8940,  676.5255,  851.6897],
                        [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
    }
]
"""

# get the box prompt for SAM 2 HQ
input_boxes = results[0]["boxes"].cpu().numpy()

# Predict masks with SAM 2 HQ
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
    hq_token_only=HQ_TOKEN_ONLY
)


"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)


confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

"""
Visualize image with supervision useful API
"""
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,
    mask=masks.astype(bool),
    class_id=class_ids,
    confidence=scores
)

"""
Note that if you want to use default color map,
you can set color=ColorPalette.DEFAULT
"""
box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

# Construct the output path based on the input image path
img_path_obj = Path(img_path)
try:
    relative_path = img_path_obj.relative_to("notebooks")
    a = relative_path.parent.name
    b = relative_path.stem
    output_filename_base = f"{a}_{b}"
except ValueError:
    # Handle cases where the image is not under 'notebooks'
    print(f"Warning: Image path '{img_path}' is not relative to 'notebooks'. Using default naming.")
    output_filename_base = Path(img_path).stem

output_filename_annotated = f"{output_filename_base}_groundingdino_annotated.jpg"
output_path_annotated = os.path.join(OUTPUT_DIR, output_filename_annotated)
cv2.imwrite(output_path_annotated, annotated_frame)

mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame_with_mask = mask_annotator.annotate(scene=img.copy(), detections=detections)

# Also update the second image filename for consistency
output_filename_mask = f"{output_filename_base}_grounded_sam2_hf_mask.jpg" # Adjusted suffix slightly for clarity
output_path_mask = os.path.join(OUTPUT_DIR, output_filename_mask)
cv2.imwrite(output_path_mask, annotated_frame_with_mask)


"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    mask_uint8 = mask.astype(np.uint8)
    rle = mask_util.encode(np.array(mask_uint8[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes_list = input_boxes.tolist()
    segmentation_scores = scores.tolist()
    detection_confidences = confidences

    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "segmentation_score": seg_score,
                "detection_confidence": det_conf,
            }
            for class_name, box, mask_rle, seg_score, det_conf in zip(class_names, input_boxes_list, mask_rles, segmentation_scores, detection_confidences)
        ],
        "box_format": "xyxy",
        "img_width": image_pil.width,
        "img_height": image_pil.height,
    }
    
    output_json_path = os.path.join(OUTPUT_DIR, "grounded_sam2_hq_v2_results.json")
    print(f"Saving results to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

print(f"Processing finished. Results saved in {OUTPUT_DIR}")
