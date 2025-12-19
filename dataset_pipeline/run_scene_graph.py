import argparse
import cv2
import glob
import math
import numpy as np
import os
import random
import sys
import time
import warnings

from typing import Any, Dict, Tuple, List

from robotic_scene_mcp.core.networkx_layer import SceneGraphNX
from robotic_scene_mcp.models import (
    BoundingBox3D,
    ObjectNode,
    Orientation,
    Position3D,
    Pose,
    Provenance,
    RelationType,
    Relation,
)

from mmengine import Config
from osdsynth.processor.captions import CaptionImage
from osdsynth.processor.pointcloud import PointCloudReconstruction
from osdsynth.processor.predicates import SpatialRelationsGenerator

from osdsynth.processor.segment_hf import SegmentImage
from osdsynth.utils.logger import (
    SkipImageException,
    save_detection_list_to_json,
    save_relations_3d_to_json,
    setup_logger,
)
from tqdm import tqdm

# Suppressing all warnings
warnings.filterwarnings("ignore")


def main(args):
    """Main function to control the flow of the program."""
    # Parse arguments
    cfg = Config.fromfile(args.config)
    exp_name = args.name if args.name else args.timestamp

    # Create log folder
    cfg.log_folder = os.path.join(args.log_dir, exp_name)
    os.makedirs(os.path.abspath(cfg.log_folder), exist_ok=True)

    # Create Wis3D folder
    cfg.vis = args.vis
    cfg.wis3d_folder = os.path.join(args.log_dir, "Wis3D")
    os.makedirs(os.path.abspath(cfg.wis3d_folder), exist_ok=True)

    # Init the logger and log some basic info
    cfg.log_file = os.path.join(cfg.log_folder, f"{exp_name}_{args.timestamp}.log")
    logger = setup_logger()  # cfg.log_file
    logger.info(f"Config:\n{cfg.pretty_text}")

    # Dump config to log
    cfg.dump(os.path.join(cfg.log_folder, os.path.basename(args.config)))

    # Create output folder
    cfg.exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(os.path.abspath(cfg.exp_dir), exist_ok=True)

    # Create folder for output json
    cfg.json_folder = os.path.join(cfg.exp_dir, "json")
    os.makedirs(os.path.abspath(cfg.json_folder), exist_ok=True)

    global_data = glob.glob(f"{args.input}/*.jpg") + glob.glob(f"{args.input}/*.png")
    device = "cuda"

    annotate(cfg, global_data, logger, device)


def annotate(cfg, global_data, logger, device):
    random.shuffle(global_data)

    segmenter = SegmentImage(cfg, logger, device)
    reconstructor = PointCloudReconstruction(cfg, logger, device)
    captioner = CaptionImage(cfg, logger, device)
    generator_3d = SpatialRelationsGenerator(cfg, logger, device)

    for i, filepath in tqdm(enumerate(global_data), ncols=25):
        filename = filepath.split("/")[-1].split(".")[0]
        print(f"Processing file: {filename}")

        progress_file_path = os.path.join(cfg.log_folder, f"{filename}.progress")
        if os.path.exists(progress_file_path) and cfg.check_exist:
            continue

        image_bgr = cv2.imread(filepath)
        image_bgr = cv2.resize(
            image_bgr, (int(640 / (image_bgr.shape[0]) * (image_bgr.shape[1])), 640)
        )

        try:
            # Run tagging model and get openworld detections
            vis_som, detection_list = segmenter.process(image_bgr)

            # Lift 2D to 3D, 3D bbox informations are included in detection_list
            detection_list = reconstructor.process(filename, image_bgr, detection_list)

            # Get LLaVA local caption for each region, however, currently just use a <region> placeholder
            detection_list = captioner.process_local_caption(detection_list)

            # Save detection list to json
            detection_list_path = os.path.join(cfg.json_folder, f"{filename}.json")
            save_detection_list_to_json(detection_list, detection_list_path)

            relations_3d = generator_3d.analyze_relations(detection_list)

            # Save detection list to json
            relations_3d_path = os.path.join(cfg.json_folder, f"{filename}_rels.json")
            save_relations_3d_to_json(relations_3d, relations_3d_path)

            scene_graph_3d = build_scene_graph(detection_list, relations_3d)

            scene_graph_3d_path = os.path.join(cfg.json_folder, f"{filename}_3dsg.json")
            with open(scene_graph_3d_path, "w") as f:
                f.write(scene_graph_3d.to_json())

        except SkipImageException as e:
            # Meet skip image condition
            logger.info(f"Skipping processing {filename}: {e}.")
            continue


def build_scene_graph(nodes, relations):
    print(f"build_scene_graph nodes: {nodes}\nrelations: {relations}", file=sys.stderr)
    graph = SceneGraphNX()
    nodes_map = {}
    for node in nodes:
        object = convert_node(node)
        nodes_map[node["id"]] = object
        graph.add_node(object)
    convert_and_add_relations(relations, nodes_map, graph)
    return graph


def convert_node(
    detection: Dict[str, Any], room_id: str = "", robot_id: str = "robot_0"
) -> ObjectNode:
    """
    Converts a raw detection dictionary into a Pydantic ObjectNode.
    """

    # BaseNode Properties
    # =====================
    # name: str = Field(..., min_length=1, max_length=256)
    # description: str = Field(default="")
    # properties: dict[str, Any] = Field(default_factory=dict)
    # provenance: Provenance | None = None
    # embedding: list[float] | None = Field(default=None, description="Semantic embedding vector")

    # ObjectNode Properties
    # =====================
    # object_class: str = Field(..., description="Object category (chair, table, etc.)")
    # room_id: str = Field(..., description="Containing room ID")
    # pose: Pose | None = None
    # bounds: BoundingBox3D | None = None
    # affordances: list[AffordanceType] = Field(default_factory=list)
    # safety_level: SafetyLevel = Field(default=SafetyLevel.SAFE)
    # is_movable: bool = Field(default=True)
    # is_interactive: bool = Field(default=False)
    # state: dict[str, Any] = Field(default_factory=dict)  # e.g., {"door": "open"}
    # confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # 1. Handle Oriented Bounding Box (Open3D Object)
    pose = None
    obb = detection.get("oriented_bbox")
    if obb:
        # Accessing Open3D attributes directly
        center = obb.center
        rot_matrix = obb.R

        pos = Position3D(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        ori = matrix_to_quaternion(np.asarray(rot_matrix))
        pose = Pose(position=pos, orientation=ori)

    # 2. Handle Axis Aligned Bounding Box (Open3D Object)
    bounds = None
    aabb = detection.get("axis_aligned_bbox")
    if aabb:
        # Accessing Open3D attributes directly
        min_b = aabb.min_bound
        max_b = aabb.max_bound
        bounds = BoundingBox3D(
            min_x=float(min_b[0]),
            max_x=float(max_b[0]),
            min_y=float(min_b[1]),
            max_y=float(max_b[1]),
            min_z=float(min_b[2]),
            max_z=float(max_b[2]),
        )

    # 3. Create Provenance
    confidence_val = float(detection.get("confidence", 1.0))
    provenance = Provenance(
        source_robot_id=robot_id, confidence=confidence_val, sensor_modality="rgb-d"
    )

    # 4. Handle Properties (stripping Open3D objects and mapped keys)
    mapped_keys = {
        "oriented_bbox",
        "axis_aligned_bbox",
        "confidence",
        "class_name",
        "label",
    }
    properties = {k: v for k, v in detection.items() if k not in mapped_keys}

    # 5. Construct ObjectNode
    return ObjectNode(
        name=detection.get("label") or detection.get("class_name", "object"),
        object_class=detection.get("class_name", "unknown"),
        room_id=room_id,
        pose=pose,
        bounds=bounds,
        confidence=confidence_val,
        provenance=provenance,
        properties=properties,
    )


def convert_and_add_relations(
    source_relations: List[Tuple[str, str, str, Any]],
    nodes_map: Dict[str, Any],
    scene_graph: SceneGraphNX,
):
    """
    Converts source relation tuples into Relation models and adds them to the graph.
    Injects semantic opposites for directed predicates to ensure graph completeness.
    """

    # Precise mapping of source keys to (Forward, Reverse) RelationType pairs
    RELATION_MAP = {
        "direction": (RelationType.DIRECTION, None),
        "left_predicate": (RelationType.LEFT_Of, RelationType.RIGHT_Of),
        "thin_predicate": (RelationType.THINNER_THAN, RelationType.LARGER_THAN),
        "small_predicate": (RelationType.SMALLER_THAN, RelationType.TALLER_THAN),
        "front_predicate": (RelationType.IN_FRONT_OF, RelationType.BEHIND),
        "below_predicate": (RelationType.BELOW, RelationType.ABOVE),
        "short_predicate": (RelationType.SHORTER_THAN, RelationType.LONGER_THAN),
        "vertical_distance_data": (RelationType.VERTICAL_DISTANCE, None),
        "horizontal_distance_data": (RelationType.HORIZONTAL_DISTANCE, None),
        "distance": (RelationType.DISTANCE, None),
    }

    # Attributes specifically requested to be ignored
    IGNORED_ATTRIBUTES = {"width_data", "height_data"}

    for src_id, tgt_id, rel_name, value in source_relations:
        # Filter out ignored attributes and non-existent relations (False/0)
        if rel_name in IGNORED_ATTRIBUTES or value is False or value == 0:
            continue

        mapping = RELATION_MAP.get(rel_name)
        if not mapping:
            continue

        forward_type, reverse_type = mapping

        # Determine numeric weight/value
        try:
            weight = float(value)
        except (ValueError, TypeError):
            weight = 1.0

        # Floats represent scalar distances; these use the bidirectional flag
        # because the distance from A to B is the same as B to A.
        is_float_measure = isinstance(value, float)

        # 1. Add the primary forward relation
        forward_rel = Relation(
            source_id=src_id["id"].item(),
            target_id=tgt_id["id"].item(),
            relation_type=forward_type,
            weight=weight,
            bidirectional=is_float_measure,
            properties={"raw_source": rel_name},
        )
        scene_graph.add_relation(forward_rel)

        # 2. Add the semantic opposite if the relation is directed (not a float distance)
        # and an opposite mapping exists in the current RelationType enum.
        if not is_float_measure and reverse_type:
            reverse_rel = Relation(
                source_id=tgt_id["id"].item(),
                target_id=src_id["id"].item(),
                relation_type=reverse_type,
                weight=weight,
                bidirectional=False,
                properties={"derived_as_opposite_of": rel_name},
            )
            scene_graph.add_relation(reverse_rel)


def matrix_to_quaternion(R: list[list[float]]) -> Orientation:
    """Matrix to Quaternion Conversion"""
    # Trace of the matrix
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return Orientation(
            w=0.25 * s,
            x=(R[2][1] - R[1][2]) / s,
            y=(R[0][2] - R[2][0]) / s,
            z=(R[1][0] - R[0][1]) / s,
        )
    elif (R[0][0] > R[1][1]) and (R[0][0] > R[2][2]):
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        return Orientation(
            w=(R[2][1] - R[1][2]) / s,
            x=0.25 * s,
            y=(R[0][1] + R[1][0]) / s,
            z=(R[0][2] + R[2][0]) / s,
        )
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        return Orientation(
            w=(R[0][2] - R[2][0]) / s,
            x=(R[0][1] + R[1][0]) / s,
            y=0.25 * s,
            z=(R[1][2] + R[2][1]) / s,
        )
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        return Orientation(
            w=(R[1][0] - R[0][1]) / s,
            x=(R[0][2] + R[2][0]) / s,
            y=(R[1][2] + R[2][1]) / s,
            z=0.25 * s,
        )


def parse_args():
    """Command-line argument parser."""
    parser = argparse.ArgumentParser(description="Generate 3D SceneGraph for an image.")
    parser.add_argument(
        "--config", default="configs/v2.py", help="Annotation config file path."
    )
    parser.add_argument(
        "--input",
        default="./demo_images",
        help="Path to input, can be json of folder of images.",
    )
    parser.add_argument(
        "--output-dir",
        default="./demo_out",
        help="Path to save the scene-graph JSON files.",
    )
    parser.add_argument(
        "--name",
        required=False,
        default=None,
        help="Specify, otherwise use timestamp as nameing.",
    )
    parser.add_argument(
        "--log-dir",
        default="./demo_out/log",
        help="Path to save logs and visualization results.",
    )
    parser.add_argument(
        "--vis",
        required=False,
        default=True,
        help="Wis3D visualization for reconstruted pointclouds.",
    )
    parser.add_argument(
        "--overwrite", required=False, action="store_true", help="Overwrite previous."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.timestamp = timestamp
    main(args)
