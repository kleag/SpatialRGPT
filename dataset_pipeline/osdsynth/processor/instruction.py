import random
from itertools import combinations
import osdsynth.processor.predicates as preds

from osdsynth.processor.instruction_template import (
    direction_responses,
    height_answers,
    width_answers,
    horizontal_distance_answers,
    vertical_distance_answers,
    front_true,
    front_false,
    small_true_responses,
    small_false_responses,
    thin_true_responses,
    thin_false_responses,
    short_true_responses,
    short_false_responses,
    below_true_responses,
    below_false_responses,
    left_true_responses,
    left_false_responses,
    distance_template_answers,
)


from osdsynth.processor.pointcloud import (
    human_like_distance,
)


def left_predicate(A, B):
    true_responses = left_true_responses
    false_responses = left_false_responses
    A_desc = A["caption"]
    B_desc = B["caption"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    is_left = preds.left_predicate(A, B)

    response_template = random.choice(true_responses if is_left else false_responses)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def below_predicate(A, B):
    true_responses = below_true_responses
    false_responses = below_false_responses
    A_desc = A["caption"]
    B_desc = B["caption"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    is_below = preds.below_predicate(A, B)

    response_template = random.choice(true_responses if is_below else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def short_predicate(A, B):
    true_responses = short_true_responses
    false_responses = short_false_responses
    A_desc = A["caption"]
    B_desc = B["caption"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    is_shorter = preds.short_predicate(A, B)

    response_template = random.choice(true_responses if is_shorter else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def thin_predicate(A, B):
    true_responses = thin_true_responses
    false_responses = thin_false_responses

    A_desc = A["caption"]
    B_desc = B["caption"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    is_thinner = preds.thin_predicate(A, B)

    response_template = random.choice(true_responses if is_thinner else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def small_predicate(A, B):
    true_responses = small_true_responses
    false_responses = small_false_responses

    A_desc = A["caption"]
    B_desc = B["caption"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    is_smaller = preds.small_predicate(A, B)

    response_template = random.choice(true_responses if is_smaller else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def front_predicate(A, B):
    true_responses = front_true
    false_responses = front_false

    A_desc = A["caption"]
    B_desc = B["caption"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    is_in_front = preds.front_predicate(A, B)

    response_template = random.choice(
        true_responses if is_in_front else false_responses
    )

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


# Distance prompts


def generate_spatial_reasoning_data(A, B, human_readable_dist, template_answers):
    A_desc, B_desc = A["caption"].lower(), B["caption"].lower()

    answer_template = random.choice(template_answers)

    # Replace placeholders with actual values
    answer = (
        answer_template.replace("[A]", A_desc)
        .replace("[B]", B_desc)
        .replace("[X]", human_readable_dist)
    )

    # Add to the dataset
    return answer


def vertical_distance_data(A, B, use_center=True):
    template_answers = vertical_distance_answers

    vertical_distance = preds.vertical_distance_data(A, B, use_center=True)

    human_readable_dist = human_like_distance(vertical_distance)

    return generate_spatial_reasoning_data(A, B, human_readable_dist, template_answers)


def distance(A, B):
    template_answers = distance_template_answers
    A_desc, B_desc = A["caption"].lower(), B["caption"].lower()
    distance = preds.distance(A, B)
    answer_template = random.choice(template_answers)

    answer = (
        answer_template.replace("[A]", A_desc)
        .replace("[B]", B_desc)
        .replace("[X]", distance)
    )
    return answer


def horizontal_distance_data(A, B, use_center=True):
    template_answers = horizontal_distance_answers

    horizontal_distance = preds.horizontal_distance_data(A, B, use_center=True)

    human_readable_dist = human_like_distance(horizontal_distance)

    return generate_spatial_reasoning_data(A, B, human_readable_dist, template_answers)


def width_data(A, B=None):
    A_desc = A["caption"].lower()

    template_answers = width_answers

    width = preds.width_data(A, B)
    human_readable_width = human_like_distance(width)

    answer_template = random.choice(template_answers)

    answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_width)

    return answer


def height_data(A, B=None):
    A_desc = A["caption"].lower()

    template_answers = height_answers

    height = preds.height_data(A, B)

    human_readable_height = human_like_distance(height)
    answer_template = random.choice(template_answers)

    answer = answer_template.replace("[A]", A_desc).replace(
        "[X]", human_readable_height
    )

    return answer


def direction(A, B):
    template_responses = direction_responses
    A_desc, B_desc = A["caption"].lower(), B["caption"].lower()

    clock_position = preds.direction(A, B)

    answer_template = random.choice(template_responses)

    answer = (
        answer_template.replace("[X]", str(int(clock_position)))
        .replace("[A]", A_desc)
        .replace("[B]", B_desc)
    )

    return answer


class PromptGenerator:
    def __init__(self, cfg, logger, device):
        """Initialize the class."""
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.vis = True

    def evaluate_predicates_on_pairs(self, detections):
        all_combinations = list(combinations(range(len(detections)), 2))
        random.shuffle(all_combinations)
        selected_combinations = all_combinations[:3]
        object_pairs = [
            (detections[i], detections[j]) for i, j in selected_combinations
        ]

        all_prompt_variants = [
            # direction,
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

            # selected_predicates_choices = all_prompt_variants
            selected_predicates_choices = random.sample(all_prompt_variants, 3)

            for prompt_func in selected_predicates_choices:
                results.append(prompt_func(A, B))

        return results
