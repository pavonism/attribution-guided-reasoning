import argparse
import glob
import os
from pathlib import Path
import traceback
from typing import Any, Dict

from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image

import src.model.mask_attribution as mask


def get_image_columns(dataset: Dataset) -> list[str]:
    image_columns = [col for col in dataset.features if col.startswith("image")]
    return sorted(image_columns)


def load_attributions(
    entities_path: str,
) -> mask.AttributionResults:
    mask_attr = mask.AttributionResults(attributions=[])

    if Path(entities_path).exists():
        entity_files = (
            [entities_path]
            if os.path.isfile(entities_path)
            else glob.glob(f"{entities_path}/**/*.json", recursive=True)
        )

        for file_path in entity_files:
            try:
                current = mask.AttributionResults.from_file(file_path)
                dir_path = Path(file_path).parent

                for problem_attr in current.attributions:
                    for img_attr in problem_attr.images:
                        for i in range(len(img_attr.mask_files)):
                            for j in range(len(img_attr.mask_files[i])):
                                if img_attr.mask_files[i][j].startswith("images/"):
                                    img_attr.mask_files[i][j] = str(
                                        dir_path / img_attr.mask_files[i][j]
                                    )

                mask_attr.attributions.extend(current.attributions)
            except Exception:
                print(f"Error: Could not load {file_path}")
                print(f"Error details: {traceback.format_exc()}")
        
            print(f"Loaded {file_path}")

    return mask_attr


def create_attribution_mappings(
    mask_attr: mask.AttributionResults,
) -> dict[tuple[str, str], mask.ImageAttributions]:
    mask_mapping = {}
    for problem_attr in mask_attr.attributions:
        for img_attr in problem_attr.images:
            key = (problem_attr.problem_id, img_attr.column_name)
            mask_mapping[key] = img_attr

    return mask_mapping


def append_attributions(
    batch: Dict[str, Any],
    mask_mapping: dict[tuple[str, str], mask.ImageAttributions],
    image_columns: list[str],
    problem_id_column: str,
) -> Dict[str, Any]:
    batch_size = len(batch[problem_id_column])

    for col in image_columns:
        col_suffix = col.replace("image_", "")
        batch[f"entities_{col_suffix}"] = [[] for _ in range(batch_size)]
        batch[f"masks_{col_suffix}"] = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        problem_id = batch[problem_id_column][i]

        for col in image_columns:
            col_suffix = col.replace("image_", "")
            key = (problem_id, col)
            if key in mask_mapping:
                mask_attr = mask_mapping[key]
                batch[f"entities_{col_suffix}"][i] = mask_attr.entities

                mask_images = []

                for curr_mask_file_paths in mask_attr.mask_files:
                    curr_mask_images = []
                    for mask_file_path in curr_mask_file_paths:
                        try:
                            img = Image.open(mask_file_path).copy()
                            curr_mask_images.append(img)
                        except Exception:
                            print(
                                f"Warning: Could not load mask image {curr_mask_file_paths}: {traceback.format_exc()}"
                            )

                    mask_images.append(curr_mask_images)

                batch[f"masks_{col_suffix}"][i] = mask_images

    return batch


def main():
    parser = argparse.ArgumentParser(
        description="Append entity and mask attributions to a HuggingFace dataset."
    )
    parser.add_argument(
        "--dataset", help="Path to the dataset directory or HuggingFace dataset name."
    )
    parser.add_argument(
        "--entities",
        help="Path to the directory containing entity and mask attribution files.",
    )
    parser.add_argument("--output", help="Path to output dataset directory.")
    parser.add_argument(
        "--problem-id-column",
        help="Name of the problem ID column in the dataset.",
        default="id",
    )
    parser.add_argument(
        "--num-proc",
        help="Number of processes for dataset.map().",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for processing.",
        type=int,
        default=10,
    )

    args = parser.parse_args()
    dataset_name_or_path = args.dataset
    attributions_path = args.entities
    output_path = args.output
    problem_id_column = args.problem_id_column
    num_proc = args.num_proc
    batch_size = args.batch_size

    print(f"Dataset: {dataset_name_or_path}")
    print(f"Attributions path: {attributions_path}")
    print(f"Output path: {output_path}")
    print(f"Problem ID column: {problem_id_column}")
    print(f"Number of processes: {num_proc}")
    print(f"Batch size: {batch_size}")

    if os.path.exists(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path, split="train")

    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Dataset columns: {dataset.column_names}")

    image_columns = get_image_columns(dataset)
    print(f"Image columns: {image_columns}")

    print("Loading attributions...")
    mask_attr = load_attributions(attributions_path)

    print("Creating attribution mappings...")
    mask_mapping = create_attribution_mappings(mask_attr)

    print("Appending attributions to dataset...")

    def append_fn(batch):
        return append_attributions(
            batch,
            mask_mapping,
            image_columns,
            problem_id_column,
        )

    updated_dataset = dataset.map(
        append_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Appending attributions",
        writer_batch_size=100,
    )

    os.makedirs(output_path, exist_ok=True)
    updated_dataset.save_to_disk(output_path, num_proc=num_proc)
    print(f"Dataset saved to: {output_path}")
    print(f"New dataset columns: {updated_dataset.column_names}")


if __name__ == "__main__":
    main()
