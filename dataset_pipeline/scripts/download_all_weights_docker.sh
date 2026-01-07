#!/bin/bash

set -o errexit

WEIGHTS_DIR=${1:-"/app/weights"}
install -d "$WEIGHTS_DIR/grounded_sam" "$WEIGHTS_DIR/perspective_fields" "$WEIGHTS_DIR/depth_anything" "$WEIGHTS_DIR/SpatialRGPT-VILA1.5-8B"

echo "Downloading weights to $WEIGHTS_DIR..."

# --- Grounded-SAM Weights ---
# Using aria2c for parallel connections (-x 16) to speed up large files
aria2c -x 16 -s 16 -d "$WEIGHTS_DIR/grounded_sam" \
    -o "sam_vit_h_4b8939.pth" \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" 
aria2c -x 16 -s 16 -d "$WEIGHTS_DIR/grounded_sam" \
    -o "sam_hq_vit_h.pth" \
    "https://huggingface.co/Uminosachi/sam-hq/resolve/main/sam_hq_vit_h.pth" 
aria2c -x 16 -s 16 -d "$WEIGHTS_DIR/grounded_sam" \
    -o "groundingdino_swint_ogc.pth" \
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# RAM (Tag2Text)
aria2c -x 16 -s 16 -d "$WEIGHTS_DIR/grounded_sam" \
    -o "ram_swin_large_14m.pth" \
    "https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth"
aria2c -x 16 -s 16 -d "$WEIGHTS_DIR/grounded_sam" \
    -o "tag2text_swin_14m.pth" \
    "https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth"
    
# --- PerspectiveFields Weights ---
aria2c -x 16 -s 16 -d "$WEIGHTS_DIR/perspective_fields" \
    -o "paramnet_360cities_edina_rpf.pth" \
    "https://www.dropbox.com/s/z2dja70bgy007su/paramnet_360cities_edina_rpf.pth"

# Depth Anything
aria2c -x 16 -s 16 -d "$WEIGHTS_DIR/depth_anything" \
    -o "depth_anything_vitl14.pth" \
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"


huggingface-cli download a8cheng/SpatialRGPT-VILA1.5-8B --repo-type model --local-dir "$WEIGHTS_DIR/SpatialRGPT-VILA1.5-8B"

touch "$WEIGHTS_DIR/DONE"
echo "Download complete."






