from math import ceil
import os
from pathlib import Path
from dataclasses import dataclass
from datasets import Dataset, load_dataset, load_from_disk
import argparse
import glob
import PIL.Image
from PIL.Image import Image
import torch
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model

import random

import numpy as np

import src.model.entity_attributions as ent
import src.model.mask_attribution as mask

from src.dataset_adapters import DatasetAdapter, AutoDatasetAdapter


@dataclass
class Index:
    value: int


def save_masks(
    masks: list[list[np.ndarray]],
    image_global_index: Index,
    output_path: str,
) -> list[list[str]]:
    mask_paths = []

    for entity_masks in masks:
        entity_masks_paths = []

        for entity_mask in entity_masks:
            mask_img_array = (entity_mask > 0).astype(np.uint8) * 255
            img = PIL.Image.fromarray(mask_img_array, mode="L")
            relative_path = f"images/{image_global_index.value:04d}.png"
            img_path = f"{output_path}/{relative_path}"
            img.save(img_path)

            entity_masks_paths.append(relative_path)
            image_global_index.value += 1

        mask_paths.append(entity_masks_paths)

    return mask_paths


def iou(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    union = np.logical_or(mask_1, mask_2).sum()
    return np.logical_and(mask_1, mask_2).sum() / union if union > 0 else 0.0


def attribute_image(
    model: Sam3Model,
    processor: Sam3Processor,
    column_name: str,
    image: Image,
    entities: list[str],
    output_path: str,
    iou_threshold: float,
    image_global_index: Index,
) -> mask.ImageAttributions:
    inputs = processor(
        images=[image] * len(entities),
        text=entities,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )

    masks = [result["masks"].cpu().numpy() for result in results]

    print(
        f"[BEFORE][{column_name}]: {[f'{e}: ({len(m)})' for e, m in zip(entities, masks)]}"
    )

    result_entities = []
    result_masks = []
    accepted_masks_pool = []

    for entity, curr_entity_masks in zip(entities, masks):
        kept_masks_for_entity = []

        for curr_mask in curr_entity_masks:
            is_duplicate = False

            for accepted_mask in accepted_masks_pool:
                if iou(curr_mask, accepted_mask) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_masks_for_entity.append(curr_mask)
                accepted_masks_pool.append(curr_mask)

        if len(kept_masks_for_entity) > 0:
            result_entities.append(entity)
            result_masks.append(kept_masks_for_entity)

    print(
        f"[AFTER][{column_name}]: {[f'{e}: ({len(m)})' for e, m in zip(result_entities, result_masks)]}"
    )
    mask_files = save_masks(result_masks, image_global_index, output_path)

    return mask.ImageAttributions(
        column_name=column_name,
        entities=result_entities,
        mask_files=mask_files,
    )


def attribute(
    model: Sam3Model,
    processor: Sam3Processor,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    entities: ent.EntityAttributionResults,
    output_path: str,
    n_problems: int,
    coverage_threshold: float,
    job_id: int,
    job_count: int,
):
    attribution_results = mask.AttributionResults(attributions=[])
    entities_by_problem_id = {p.problem_id: p for p in entities.attributions}

    if Path(f"{output_path}/attributions.json").exists():
        print(f"Resuming evaluation from {output_path}")
        attribution_results = mask.AttributionResults.from_directory(output_path)

    images_dir = Path(output_path, "images")
    images_dir.mkdir(parents=True, exist_ok=True)

    all_problem_ids = dataset_adapter.get_problem_ids(dataset)

    problems_ids_to_attribute = all_problem_ids[:n_problems]

    problem_ids_per_job = ceil(len(problems_ids_to_attribute) / job_count)
    start_index = (job_id - 1) * problem_ids_per_job
    job_problem_indexes = list(
        range(
            start_index,
            min(start_index + problem_ids_per_job, len(problems_ids_to_attribute)),
        )
    )

    attributed_problem_ids = set(a.problem_id for a in attribution_results.attributions)

    current_img_index = max([int(Path(f).stem) for f in images_dir.glob("*.png")] + [0])
    image_global_index = Index(current_img_index)

    for index in tqdm(job_problem_indexes):
        problem_id = all_problem_ids[index]

        if problem_id in attributed_problem_ids:
            print(f"Skipping already attributed problem ID: {problem_id}")
            continue

        problem_entities = entities_by_problem_id.get(all_problem_ids[index])
        batch = dataset[index : index + 1]

        problem_attr = mask.ProblemAttributions(
            problem_id=problem_id,
            images=[],
        )

        for column_name, image in zip(
            dataset_adapter.get_image_columns(),
            dataset_adapter.get_images(batch)[0],
        ):
            img_entities = [
                e for e in problem_entities.images if e.column_name == column_name
            ][0].entities

            if not img_entities:
                continue

            attribution = attribute_image(
                model,
                processor,
                column_name,
                image,
                img_entities,
                output_path,
                coverage_threshold,
                image_global_index,
            )
            problem_attr.images.append(attribution)

        attribution_results.attributions.append(problem_attr)
        attribution_results.to_file(Path(output_path, "attributions.json"))


def main():
    parser = argparse.ArgumentParser(description="Attribute mask using SAM3.")
    parser.add_argument("--dataset", help="Path to test dataset.")
    parser.add_argument(
        "--entities",
        help="Path to the entity file or directory containing entity files.",
    )
    parser.add_argument("--output", help="Path to attribution output directory.")
    parser.add_argument(
        "--n_problems",
        help="Number of problems to attribute.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility.",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--coverage_threshold",
        help="Coverage threshold for mask filtering.",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--job-id",
        help="Job ID for array jobs.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--job-count",
        help="Total number of jobs in the array.",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    dataset_name_or_path = args.dataset
    entities_path = args.entities
    output_path = args.output
    seed = args.seed
    n_problems = args.n_problems
    seed = args.seed
    coverage_threshold = args.coverage_threshold
    job_id = args.job_id
    job_count = args.job_count

    print(f"Dataset: {dataset_name_or_path}")
    print(f"Output path: {output_path}")
    print(f"Entities path: {entities_path}")
    print(f"Number of problems: {n_problems}")
    print(f"Seed: {seed}")
    print(f"Coverage Threshold: {coverage_threshold}")
    print(f"Job ID: {job_id}")
    print(f"Job count: {job_count}")

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    if os.path.exists(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path, split="test")

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        dataset_name_or_path,
        dataset,
        text_only=False,
    )

    entities = ent.EntityAttributionResults(attributions=[])
    attr_files = (
        [entities_path]
        if os.path.isfile(entities_path)
        else glob.glob(f"{entities_path}/*.json")
    )

    for file_path in attr_files:
        current = ent.EntityAttributionResults.from_file(file_path)
        entities.attributions.extend(current.attributions)

    if job_count > 1:
        output_path = f"{output_path}/{job_id}_of_{job_count}"

    attribute(
        model,
        processor,
        dataset,
        dataset_adapter,
        entities,
        output_path,
        n_problems,
        coverage_threshold,
        job_id,
        job_count,
    )


if __name__ == "__main__":
    main()
