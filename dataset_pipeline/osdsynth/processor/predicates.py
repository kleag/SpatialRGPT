import random
import numpy as np
from itertools import combinations
import json

from osdsynth.processor.prompt_utils import (
    generate_random_string,
    calculate_angle_clockwise,
    is_aligned_vertically,
    is_aligned_horizontally,
    is_y_axis_overlapped,
    is_supporting,
)

from osdsynth.processor.pointcloud import (
    calculate_distances_between_point_clouds,
)


def left_predicate(A, B):
    A_cloud = A["pcd"]
    B_cloud = B["pcd"]

    A_pos = A_cloud.get_center()
    B_pos = B_cloud.get_center()

    is_left = A_pos[0] > B_pos[0]  # Compare X coordinates

    return is_left


def below_predicate(A, B):
    A_cloud = A["pcd"]
    B_cloud = B["pcd"]

    A_pos = A_cloud.get_center()
    B_pos = B_cloud.get_center()

    is_below = A_pos[1] < B_pos[1]

    return is_below


def short_predicate(A, B):
    A_cloud = A["pcd"]
    B_cloud = B["pcd"]

    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]

    is_shorter = height_A < height_B

    return is_shorter


def thin_predicate(A, B):
    A_cloud = A["pcd"]
    B_cloud = B["pcd"]

    width_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[0]
    width_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[0]

    is_thinner = width_A < width_B

    return is_thinner


def small_predicate(A, B):
    A_cloud = A["pcd"]
    B_cloud = B["pcd"]

    extent_A = A_cloud.get_axis_aligned_bounding_box().get_extent()
    volume_A = extent_A[0] * extent_A[1] * extent_A[2]

    extent_B = B_cloud.get_axis_aligned_bounding_box().get_extent()
    volume_B = extent_B[0] * extent_B[1] * extent_B[2]

    is_smaller = volume_A < volume_B

    return is_smaller


def front_predicate(A, B):
    A_cloud = A["pcd"]
    B_cloud = B["pcd"]

    # Calculate the minimum z-value for both A and B
    A_min_z = A_cloud.get_min_bound()[2]
    B_min_z = B_cloud.get_min_bound()[2]
    # Determine if A is behind B based on the minimum z-value
    is_in_front = A_min_z < B_min_z

    return is_in_front


# Distance prompts


def vertical_distance_data(A, B, use_center=True):
    # Get the bounding boxes for both A and B
    A_box = A["pcd"].get_axis_aligned_bounding_box()
    B_box = B["pcd"].get_axis_aligned_bounding_box()

    if use_center:
        A_center = A_box.get_axis_aligned_bounding_box().get_center()
        B_center = B_box.get_axis_aligned_bounding_box().get_center()
        vertical_distance = abs(A_center[1] - B_center[1])
    else:
        # Determine the highest and lowest points (in terms of y-value) of each object
        A_min_y, A_max_y = A_box.get_min_bound()[1], A_box.get_max_bound()[1]
        B_min_y, B_max_y = B_box.get_min_bound()[1], B_box.get_max_bound()[1]

        # Assuming A is above B, adjust if it's the other way around
        if A_min_y < B_min_y:
            # This means B is above A, swap the values
            A_min_y, A_max_y, B_min_y, B_max_y = B_min_y, B_max_y, A_min_y, A_max_y

        # The vertical distance is now the difference between the lowest point of the higher object (B_max_y)
        # and the highest point of the lower object (A_min_y), considering A is below B after the possible swap.
        vertical_distance = A_min_y - B_max_y if A_min_y > B_max_y else 0

    return vertical_distance


def distance(A, B):
    distance = calculate_distances_between_point_clouds(A["pcd"], B["pcd"])
    return distance


def horizontal_distance_data(A, B, use_center=True):
    # Extract bounding boxes for A and B
    A_box = A["pcd"].get_axis_aligned_bounding_box()
    B_box = B["pcd"].get_axis_aligned_bounding_box()

    if use_center:
        A_center = A_box.get_center()
        B_center = B_box.get_center()
        horizontal_distance = np.sqrt((A_center[0] - B_center[0]) ** 2)
    else:
        # Extract min and max bounds for A and B on x and z axes
        A_min, A_max = A_box.get_min_bound(), A_box.get_max_bound()
        B_min, B_max = B_box.get_min_bound(), B_box.get_max_bound()

        # Calculate the shortest horizontal (x, z plane) distance between the two boxes
        horizontal_distance = max(A_min[0] - B_max[0], B_min[0] - A_max[0], 0)

    return horizontal_distance


def width_data(A, B=None):
    width = A["pcd"].get_axis_aligned_bounding_box().get_extent()[0]
    return width


def height_data(A, B=None):
    height = A["pcd"].get_axis_aligned_bounding_box().get_extent()[1]
    return height


def direction(A, B):
    A_cloud = A["pcd"]
    B_cloud = B["pcd"]

    A_pos = (A_cloud.get_center()[0], A_cloud.get_center()[2])  # Only x, z
    B_pos = (B_cloud.get_center()[0], B_cloud.get_center()[2])  # Only x, z

    clock_position = calculate_angle_clockwise(A_pos, B_pos)

    return clock_position


class SpatialRelationsGenerator:
    def __init__(self, cfg, logger, device):
        """Initialize the class."""
        pass

    def evaluate_predicates_on_pairs(self, detections):
        all_combinations = list(combinations(range(len(detections)), 2))
        # random.shuffle(all_combinations)
        selected_combinations = all_combinations  # [:3]
        object_pairs = [
            (detections[i], detections[j]) for i, j in selected_combinations
        ]

        all_prompt_variants = [
            direction,
            left_predicate,
            thin_predicate,
            small_predicate,
            front_predicate,
            below_predicate,
            short_predicate,
            vertical_distance_data,
            horizontal_distance_data,
            width_data,
            height_data,
            distance,
        ]

        results = []

        for A, B in object_pairs:
            to_remove = set()  # A set to hold items to remove

            # Remove all items in `to_remove` from `all_prompt_variants`, if present
            all_prompt_variants = [
                item for item in all_prompt_variants if item not in to_remove
            ]

            selected_predicates_choices = all_prompt_variants
            # selected_predicates_choices = random.sample(all_prompt_variants, 3)

            for prompt_func in selected_predicates_choices:
                results.append((A, B, prompt_func.__name__, prompt_func(A, B)))

        return results
