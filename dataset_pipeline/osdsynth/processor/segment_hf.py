import cv2
import shortuuid
import sys
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from mmengine import Config

# ----------------------------------------------------------------------
# 1. CORE HUGGING FACE / SEGMENT ANYTHING IMPORTS
# ----------------------------------------------------------------------
# For Grounding DINO
from transformers import (
    GroundingDinoProcessor, 
    GroundingDinoForObjectDetection,
)
# For SAM
from segment_anything import SamPredictor, sam_model_registry
# For RAM/Tagging (Using a common model, usually requires custom inference code)
from transformers import AutoProcessor, AutoModelForCausalLM # Example for a VLM for tagging

from osdsynth.processor.wrappers.ram import get_tagging_model, run_tagging_model
from osdsynth.processor.wrappers.sam import (
    convert_detections_to_dict,
    convert_detections_to_list,
    crop_detections_with_xyxy,
    filter_detections,
    get_sam_predictor,
    get_sam_segmentation_from_xyxy,
    mask_subtract_contained,
    post_process_mask,
    sort_detections_by_area,
)
from osdsynth.utils.logger import SkipImageException
from osdsynth.visualizer.som import draw_som_on_image

# Helper function to convert List[Dict] back to Supervision-style Dict[str, np.ndarray]
# Helper function to convert List[Dict] back to Supervision-style Dict[str, np.ndarray]
def reconstruct_detections_dict(detections_list: List[Dict[str, Any]], all_classes: List[str]) -> Dict[str, Any]:
    """
    Converts a List[Dict] of individual detections back into a dictionary 
    where keys map to NumPy arrays of all detections, matching the structure 
    expected by osdsynth utilities like filter_detections.
    """
    if not detections_list:
        # Handle empty case: Ensure 'classes' is still included
        return {
            "id": np.array([]), 
            "xyxy": np.array([]).reshape(0, 4), 
            "confidence": np.array([]), 
            "class_id": np.array([]), 
            "class_name": np.array([]), 
            "mask": np.array([]),
            "label": [], 
            "classes": all_classes, 
            "subtracted_mask": np.array([]),
            "area": np.array([]),
            "rle": np.array([]),
        }

    detections_dict = {
        "id": np.array([d["id"] for d in detections_list]),
        "xyxy": np.array([d["xyxy"] for d in detections_list]),
        "confidence": np.array([d["confidence"] for d in detections_list]),
        "class_id": np.array([d["class_id"] for d in detections_list]).astype(int),
        "class_name": np.array([d["class_name"] for d in detections_list]),
        "mask": np.array([d["mask"] for d in detections_list]),
        "label": [d["label"] for d in detections_list], 
        "classes": all_classes, 
        "subtracted_mask": np.array([d["subtracted_mask"] if "subtracted_mask" in d else None for d in detections_list]), 
        "area": np.array([d["area"] if "area" in d else None for d in detections_list]),
        "rle": np.array([d["rle"] if "rle" in d else None for d in detections_list]), 
    }
    return detections_dict


# We also need a reverse function, since filter_detections returns a Dict but 
# crop_detections_with_xyxy expects a List[Dict].
def convert_detections_dict_to_list(detections_dict: Dict[str, np.ndarray], all_classes: List[str]) -> List[Dict[str, Any]]:
    """Converts a Supervision-style Dict[str, np.ndarray] back to a List[Dict]."""
    if detections_dict["xyxy"].shape[0] == 0:
        return []
        
    detections_list = []
    num_dets = detections_dict["xyxy"].shape[0]
    
    # Use 'label' if available, otherwise use 'class_id' to look up in all_classes
    labels = detections_dict.get("label")
    
    for i in range(num_dets):
        class_id = detections_dict["class_id"][i]
        
        det = {
            "id": detections_dict.get("id")[i],
            "xyxy": detections_dict["xyxy"][i],
            "confidence": detections_dict["confidence"][i],
            "class_id": class_id,
            "class_name": detections_dict["class_name"][i],
            "label": labels[i] if labels and len(labels) == num_dets else all_classes[class_id],
            "mask": detections_dict["mask"][i],
            # Use .get() to safely include optional keys created by utilities
            "subtracted_mask": detections_dict.get("subtracted_mask", detections_dict["mask"])[i], 
            "area": detections_dict.get("area", 0)[i],
            "rle": detections_dict.get("rle"), # RLE is usually stored once or computed later
            # ... and any other keys added by your utility functions
        }
        detections_list.append(det)
        
    return detections_list
# ----------------------------------------------------------------------
# 2. UTILITY STUBS AND ADAPTERS
# ----------------------------------------------------------------------

# Helper function to convert HF DINO output to a standard list of dicts
def convert_gdino_output_to_detections_list(boxes, scores, labels, class_names, image_size) -> List[Dict[str, Any]]:
    """Converts normalized Grounding DINO output (boxes 0-1000) to a list of dicts with pixel coordinates."""
    print(f"convert_gdino_output_to_detections_list {labels}, {class_names}", file=sys.stderr)
    detections_list = []
    width, height = image_size
    tensor_scale = torch.Tensor([width, height, width, height]).to(boxes.device)
    
    for box, score, label_idx in zip(boxes, scores, labels):
        # Convert normalized [0, 1000] boxes to actual pixel values
        box_unnormalized = (box * tensor_scale / 1000).cpu().numpy().astype(int)
        
        detections_list.append({
            "id": shortuuid.uuid(),
            "xyxy": box_unnormalized, # Pixel coordinates [x1, y1, x2, y2]
            "confidence": score.item(),
            "class_id": class_names.index(label_idx.split(" ")[0]),
            "class_name": label_idx.split(" ")[0],
            "label": label_idx,
            "mask": None, # Placeholder for SAM output
        })
    return detections_list

# Rewritten adapter function to match the original iterative logic
def get_sam_segmentation_from_xyxy_hf(sam_predictor: SamPredictor, image_rgb: np.ndarray, detections_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Segments the image using SAM by processing each bounding box iteratively, 
    mirroring the original osdsynth approach for reliability.
    """
    if not detections_list:
        return []
    
    # 1. Extract all boxes from the list of dictionaries
    # input_boxes will be shape (N, 4) where N=18
    input_boxes = np.array([d["xyxy"] for d in detections_list])
    
    # Set the image once
    sam_predictor.set_image(image_rgb)
    
    result_masks_list = []

    # 2. Iterate and call sam_predictor.predict() for EACH SINGLE box
    for i, box in enumerate(input_boxes):
        # 'box' is now a single bounding box array (4,)
        # Use the original sam_predictor.predict() method
        masks, scores, logits = sam_predictor.predict(
            box=box, # Single box, shape (4,)
            multimask_output=True # Match original setting to get 3 masks
        )
        
        # Select the highest scoring mask (as done in the original code)
        index = np.argmax(scores)
        
        # Append the best mask (boolean array: H, W)
        result_masks_list.append(masks[index])
        
        # Attach the mask to the corresponding detection in the list
        detections_list[i]["mask"] = masks[index]
        
    return detections_list


# NOTE: In your main SegmentImage.process() call, you must revert to using 
# the original method signature (predict_torch() is removed):

# File "/scratch/gdechalendar/Projets/Ridder/SpatialRGPT/dataset_pipeline/osdsynth/processor/segment_hf.py", line 267
# ...
# 5. Segment Anything (SAM) (Replaces get_sam_segmentation_from_xyxy)
# detections_list = get_sam_segmentation_from_xyxy_hf(
#     sam_predictor=self.sam_predictor, image=image_rgb, detections_list=detections_list
    

# ----------------------------------------------------------------------
# 3. INITIALIZATION WRAPPERS (Replaces osdsynth wrappers)
# ----------------------------------------------------------------------

def get_tagging_model_hf(cfg, device):
    """
    Replaces get_tagging_model. Loads a Hugging Face tagging model (RAM/Tag2Text).
    NOTE: RAM often requires custom repo cloning, so we use a standard VLM pattern 
    for illustration, but the user may need custom loading for the exact RAM model.
    """
    try:
        # We assume the model is wrapped for simple HF loading for a clean rewrite
        processor = AutoProcessor.from_pretrained(cfg.tagging_model_id)
        model = AutoModelForCausalLM.from_pretrained(cfg.tagging_model_id).to(device)
        return processor, model
    except Exception as e:
        print(f"Warning: Could not load {cfg.tagging_model_id} via AutoModel. Using mock tagging.")
        # If model loading fails, return None/None to indicate mock usage
        return None, None

def run_tagging_model_hf(cfg, img_pil: Image.Image, tagging_transform, tagging_model) -> List[str]:
    """
    Replaces run_tagging_model. Performs image tagging to get classes.
    """
    if tagging_model is None:
        # Fallback to hardcoded list if the model failed to load
        return ["person", "car", "building", "tree"] 
        
    # Standard VLM inference pattern for tagging/captioning
    # This step should be implemented to extract tags from the VLM's output.
    # The original RAM model specifically returns a list of tags.
    
    # Placeholder for actual RAM inference logic
    # The RAM model typically outputs a string of tags separated by | or ,
    tag_string = "person, car, building, tree, road"
    classes = [tag.strip() for tag in tag_string.split(',') if tag.strip()]
    
    return classes
    
# ----------------------------------------------------------------------
# 4. REWRITTEN MAIN CLASS
# ----------------------------------------------------------------------

class SegmentImage:
    """
    Class to segment an image using Hugging Face Grounding DINO and official 
    SAM instead of manually downloaded code and models.
    """

    def __init__(self, cfg: Config, logger: Any, device: str, 
                 init_gdino=True, init_tagging=True, init_sam=True):
        """
        Constructor

        Parameters:
            cfg: config
            logger: a logger
            device: the device to run on ("cpu" or "cuda")
        """
        self.cfg = cfg
        self.logger = logger
        self.device = device

        if init_gdino:
            # Initialize the Hugging Face Grounding DINO Model
            self.gdino_processor = GroundingDinoProcessor.from_pretrained(cfg.gdino_variant)
            self.grounding_dino_model = GroundingDinoForObjectDetection.from_pretrained(cfg.gdino_variant).to(device)
        else:
            self.gdino_processor = self.grounding_dino_model = None

        if init_tagging:
            # Initialize the tagging Model (RAM/Tag2Text replacement)
            # Uncomment if the original ram version do not work
            # self.tagging_transform, self.tagging_model = get_tagging_model_hf(cfg, device)
            self.tagging_transform, self.tagging_model = get_tagging_model(cfg, device)
        else:
            self.tagging_transform = self.tagging_model = None

        if init_sam:
            # Initialize the official SAM Model
            # Try the version below if the original one do not work
            #sam = sam_model_registry[cfg.sam_variant](checkpoint=cfg.sam_checkpoint_path).to(device)
            #self.sam_predictor = SamPredictor(sam)
            self.sam_predictor = get_sam_predictor(cfg.sam_variant, device)
        else:
            self.sam_predictor = None

        pass

    def process(self, image_bgr: np.ndarray, plot_som: bool = True):
        """Segment the image."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_pil = Image.fromarray(image_rgb)
        
        # 1. Tag2Text (RAM) -> Get classes
        # Note: The original resizing/transform is handled inside run_tagging_model_hf replacement
        # Use version below (need to be implemented) if the original one does not work
        # classes = run_tagging_model_hf(self.cfg, image_rgb_pil, self.tagging_transform, self.tagging_model)

        img_tagging = image_rgb_pil.resize((384, 384))
        img_tagging = self.tagging_transform(img_tagging).unsqueeze(0).to(self.device)

        # Tag2Text
        classes = run_tagging_model(self.cfg, img_tagging, self.tagging_model)

        if len(classes) == 0:
            raise SkipImageException("No foreground objects detected by tagging model.")
        
        # Create the Grounding DINO text prompt: "class1 . class2 . class3 ."
        text_prompt = " . ".join(classes) + " ."
        
        # 2. Grounding DINO Detection (Replaces predict_with_classes)
        inputs = self.gdino_processor(images=image_rgb_pil, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_dino_model(**inputs)

        # Post-process DINO outputs and apply thresholds
        results_with_labels = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
            target_sizes=[image_rgb_pil.size[::-1]] # [H, W]
        )
        
        results = results_with_labels[0]
        boxes_normalized = results["boxes"] # tensor, normalized to 0-1000
        scores = results["scores"] # tensor
        labels = results["labels"] # tensor (class index from the 'classes' list)
        print(f"labels: {labels}")

        if len(boxes_normalized) < 1:
            raise SkipImageException("No object detected by Grounding DINO.")

        # 3. Non-maximum suppression (NMS)
        nms_idx = torchvision.ops.nms(
            boxes_normalized,
            scores,
            self.cfg.nms_threshold,
        )
        print(f"nms_idx: {nms_idx}")

        print(f"Before NMS: {len(boxes_normalized)} detections:")
        #print(f"Before NMS: detections: {boxes_normalized}")
        
        # Apply NMS indices to the tensors
        boxes_normalized_nms = boxes_normalized[nms_idx]
        scores_nms = scores[nms_idx]
        nms_list = nms_idx.cpu().numpy().tolist()
        labels_nms = [labels[i] for i in nms_list]

        print(f"After NMS: {len(boxes_normalized_nms)} detections")

        # 4. Convert DINO output to detections_list (replaces custom Detections object)
        detections_list = convert_gdino_output_to_detections_list(
            boxes_normalized_nms, scores_nms, labels_nms, classes, image_rgb_pil.size
        )
        
        # 5. Segment Anything (SAM) (Replaces get_sam_segmentation_from_xyxy)
        detections_list = get_sam_segmentation_from_xyxy_hf(
            sam_predictor=self.sam_predictor, 
            image_rgb=image_rgb, 
            detections_list=detections_list
        )
        #print(f"After SAM: detections_list={detections_list}", file=sys.stderr)
        #detections.mask = get_sam_segmentation_from_xyxy(
        #    sam_predictor=self.sam_predictor, image=image_rgb, xyxy=detections.xyxy
        #)
        detections_dict = reconstruct_detections_dict(detections_list, classes)
        # NOTE: The subsequent steps now operate on detections_list (List[Dict[str, Any]])
        
        # 6. Post-processing
        # Filter out the objects based on various criteria
        detections_dict = filter_detections(self.cfg, detections_dict, image_rgb)

        if len(detections_dict["xyxy"]) < 1:
            raise SkipImageException("No object detected after filtering.")

        # Subtract the mask of bounding boxes that are contained by it
        detections_dict["subtracted_mask"], mask_contained = mask_subtract_contained(
            detections_dict["xyxy"], detections_dict["mask"], th1=0.05, th2=0.05
        )

        # Determine which mask array to use (use subtracted_mask if the key exists)
        mask_key = "subtracted_mask" if "subtracted_mask" in detections_dict else "mask"

        # Calculate the area (sum of True pixels) for every mask in the array
        mask_areas = np.sum(detections_dict[mask_key], axis=(1, 2))

        # Insert the calculated areas into the dictionary
        detections_dict["area"] = mask_areas

        # Sort the dets by area
        detections_dict = sort_detections_by_area(detections_dict)

        # Add RLE to dict
        detections_dict = post_process_mask(detections_dict)

        # 7. Reverse Fix: Convert back to List[Dict] format for final output (and cropping)
        detections_list = convert_detections_dict_to_list(detections_dict, classes)

        # Convert the detection to a list. Each element is a dict (Already done)
        # detections_list is already in the final format

        detections_list = crop_detections_with_xyxy(self.cfg, image_rgb_pil, detections_list)

        final_detections_dict = reconstruct_detections_dict(
            detections_list, 
            classes)
        if plot_som:
            # Visualize with SoM
            # NOTE: draw_som_on_image must be adapted for List[Dict] structure
            vis_som = draw_som_on_image(
                final_detections_dict,
                image_rgb,
                label_mode="1",
                alpha=0.4,
                anno_mode=["Mask", "Mark", "Box"],
            )
        else:
            vis_som = None

        return vis_som, detections_list