import argparse
import os
from typing import Any, Dict

from PIL import Image

import datasets
from tqdm import tqdm
from src.model.bongard import (
    BongardDatasetInfo,
    BongardDatasetMetadata,
    BongardImageMetadata,
    BongardProblem,
)

PROBLEM = """
The goal in solving a Bongard Problem is to identify a concept that differentiates the left and right sides. 
All images belonging to the LEFT side represent a common, shared concept which is not present in any image from the RIGHT side, 
and vice versa - all images belonging to the RIGHT side represent a common, shared concept which is not present in any image from the LEFT side.

Solve the provided Bongard Problem and provide the concept for the LEFT side. 
Output the thinking process in <think> tags and the final answer in <answer> tags.

The output answer format should be as follows: <think> ... </think> <answer>...</answer>
""".strip()


def get_concept_counts(dataset: BongardDatasetInfo) -> Dict[str, int]:
    concept_counts: Dict[str, int] = {}
    for problem in dataset.problems:
        concept = problem.left_concept

        concept_counts[concept] = concept_counts.get(concept, 0) + 1

    return concept_counts


def add_side_images(
    problem: BongardProblem,
    data_row: Dict[str, object],
):
    for side, images in [
        ("left", problem.left_images),
        ("right", problem.right_images),
    ]:
        for index, img in enumerate(images):
            img_data = Image.open(img).copy()

            data_row[f"image_{side}_{index}"] = img_data
    return data_row


def add_whole_image(problem: BongardProblem, data_row: Dict[str, object]):
    data_row["image"] = Image.open(problem.whole_image).copy()


def add_captions(
    problem: BongardProblem,
    metadata: Dict[str, BongardImageMetadata],
    data_row: Dict[str, object],
):
    for side, images in [
        ("left", problem.left_images),
        ("right", problem.right_images),
    ]:
        for index, img in enumerate(images):
            img_metadata = metadata.get(img)

            if img_metadata is None or img_metadata.caption == "":
                print(f"WARN: Empty caption for image: {img}")
            else:
                data_row[f"caption_{side}_{index}"] = img_metadata.caption


def dataset_generator(
    args: list[Dict[str, object]],
):
    args = args[0]
    input_dir: str = args["input_dir"]
    shard_index: int = args["shard_index"]
    num_shards: int = args["num_shards"]
    multi_image: bool = args["multi_image"]
    with_captions: bool = args["with_captions"]

    print("Starting worker ", shard_index)

    dataset_info = BongardDatasetInfo.from_directory(input_dir)
    img_to_metadata = BongardDatasetMetadata.from_directory(input_dir).to_dict()
    concept_counts = get_concept_counts(dataset_info)
    shard_size = (len(dataset_info.problems) + num_shards - 1) // num_shards

    dataset_info.problems = dataset_info.problems[
        shard_index * shard_size : (shard_index + 1) * shard_size
    ]

    for problem in tqdm(dataset_info.problems):
        data_row = {}

        data_row["id"] = problem.id
        data_row["problem"] = PROBLEM
        data_row["solution"] = problem.left_concept

        if multi_image:
            add_side_images(problem, data_row)
        else:
            add_whole_image(problem, data_row)

        if with_captions:
            add_captions(problem, img_to_metadata, data_row)

        data_row["weight"] = 1.0 / concept_counts.get(problem.left_concept, 1)
        data_row["dataset"] = problem.dataset if problem.dataset is not None else ""
        data_row["dataset_problem_id"] = problem.dataset_problem_id

        yield data_row


def create_image_features(multi_image: bool) -> Dict[str, Any]:
    features = {}
    if multi_image:
        for side in ["left", "right"]:
            for index in range(6):
                features[f"image_{side}_{index}"] = datasets.Image()
    else:
        features["image"] = datasets.Image()
    return features


def create_caption_features(with_captions: bool) -> Dict[str, Any]:
    features = {}
    if with_captions:
        for side in ["left", "right"]:
            for index in range(6):
                features[f"caption_{side}_{index}"] = datasets.Value("string")
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Bongard data in Hugging Face Dataset format."
    )
    parser.add_argument("--input", help="Path to the dataset directory.")
    parser.add_argument("--output", help="Path to the output directory.")
    parser.add_argument(
        "--multi-image",
        help="Store each problem sub-image in a separate column.",
        action="store_true",
    )
    parser.add_argument(
        "--with-captions",
        help="Store image captions.",
        action="store_true",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of shards.",
    )

    args = parser.parse_args()

    input = args.input
    output = args.output
    multi_image = args.multi_image
    with_captions = args.with_captions
    num_shards = args.num_shards

    if not os.path.isdir(input):
        parser.error(f"Input directory '{input}' does not exist or is not a directory.")

    print(f"Input directory: {input}")
    print(f"Output directory: {output}")
    print(f"Multi-image: {multi_image}")
    print(f"With captions: {with_captions}")
    print(f"Number of shards: {num_shards}")

    os.makedirs(output, exist_ok=True)

    hf_dataset = datasets.Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "args": [
                {
                    "input_dir": input,
                    "shard_index": index,
                    "num_shards": num_shards,
                    "multi_image": multi_image,
                    "with_captions": with_captions,
                }
                for index in range(num_shards)
            ]
        },
        features=datasets.Features(
            {
                "id": datasets.Value("int32"),
                "problem": datasets.Value("string"),
                "solution": datasets.Value("string"),
                "weight": datasets.Value("float32"),
                "dataset": datasets.Value("string"),
                "dataset_problem_id": datasets.Value("int32"),
            }
            | create_image_features(multi_image)
            | create_caption_features(with_captions)
        ),
        cache_dir=output,
        num_proc=num_shards,
    )

    hf_dataset.save_to_disk(output)
    print(f"Dataset saved to: {output}")


if __name__ == "__main__":
    main()
