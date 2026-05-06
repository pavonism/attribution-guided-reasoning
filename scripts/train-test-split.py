import argparse
from collections import Counter
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.model.bongard import BongardDatasetInfo, BongardDatasetMetadata
from src.model.groups import PROBLEM_TO_GROUP


def get_concept_to_image_mapping(
    dataset_info: BongardDatasetInfo,
) -> dict[str, set[str]]:
    concept_to_images: dict[str, set[str]] = {}

    for problem in dataset_info.problems:
        left_concept = problem.left_concept
        right_concept = problem.right_concept
        concept = f"{left_concept} ||| {right_concept}"

        if concept not in concept_to_images:
            concept_to_images[concept] = set()

        for img in problem.left_images + problem.right_images:
            concept_to_images[concept].add(img)

    return concept_to_images


def get_concept_to_source_id_mapping(
    dataset_info: BongardDatasetInfo,
) -> dict[int, str]:
    concept_to_source_id: dict[int, str] = {}

    for problem in dataset_info.problems:
        source_problem_id = problem.id % 1_000
        left_concept = problem.left_concept
        right_concept = problem.right_concept
        concept = f"{left_concept} ||| {right_concept}"
        concept_to_source_id[concept] = source_problem_id

    return concept_to_source_id


def pick_concepts_with_unique_images(
    dataset_info: BongardDatasetInfo,
    concepts: list[str],
) -> list[str]:
    concept_to_images = get_concept_to_image_mapping(dataset_info)
    selected_concepts = []

    for concept in concepts:
        images = concept_to_images[concept]
        is_disjoint = True

        for other_concept, other_images in concept_to_images.items():
            if other_concept == concept:
                continue

            if not images.isdisjoint(other_images):
                is_disjoint = False
                break

        if is_disjoint:
            selected_concepts.append(concept)

    return selected_concepts


def organize_concepts_by_concept_group(
    concepts: list[str],
    concept_to_source_id: dict[str, int],
) -> dict[str, list[int]]:
    group_to_concepts: dict[str, list[int]] = {}

    for concept in concepts:
        problem = concept_to_source_id[concept]
        group = PROBLEM_TO_GROUP[problem]

        if group not in group_to_concepts:
            group_to_concepts[group] = []

        group_to_concepts[group].append(concept)

    return group_to_concepts


def train_test_split(
    dataset_info: BongardDatasetInfo,
    test_size: int,
) -> tuple[list[str], list[str]]:
    concept_to_source_id = get_concept_to_source_id_mapping(dataset_info)
    unique_image_concepts = pick_concepts_with_unique_images(
        dataset_info, list(concept_to_source_id.keys())
    )
    group_to_concepts = organize_concepts_by_concept_group(
        unique_image_concepts, concept_to_source_id
    )
    print("Available groups and their concepts: ", group_to_concepts)

    group_to_concepts = {k: v for k, v in group_to_concepts.items() if len(v) >= 2}
    weights = [len(concepts) for concepts in group_to_concepts.values()]
    groups = list(group_to_concepts.keys())

    selected_groups = np.random.choice(
        groups, p=np.array(weights) / sum(weights), size=test_size
    )

    test_concepts = []
    for group, count in Counter(selected_groups).items():
        concepts = group_to_concepts[group]
        selected_concepts = np.random.choice(concepts, size=count, replace=False)
        test_concepts.extend(selected_concepts.tolist())

    train_concepts = [p for p in unique_image_concepts if p not in test_concepts]
    print("Train concepts: ", train_concepts)
    print("Test concepts: ", test_concepts)
    return train_concepts, test_concepts


def copy_problems_by_concepts(
    dataset_info: BongardDatasetInfo,
    concepts: list[str],
    source_path: str,
    output_path: str,
):
    selected_problems = [
        problem
        for problem in dataset_info.problems
        if f"{problem.left_concept} ||| {problem.right_concept}" in concepts
    ]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    subset_dataset_info = BongardDatasetInfo(problems=selected_problems)
    subset_dataset_info.to_file(Path(output_path, "dataset.json"))

    split_images = []
    for problem in tqdm(selected_problems):
        imgs = (
            problem.left_images
            + problem.right_images
            + [problem.whole_image, problem.left_side_image, problem.right_side_image]
        )
        split_images.extend(imgs)

        for img in imgs:
            source_img_path = Path(source_path, img)
            dest_img_path = Path(output_path, img)

            dest_img_path.parent.mkdir(parents=True, exist_ok=True)
            #shutil.copy(source_img_path, dest_img_path)

    metadata = BongardDatasetMetadata.from_directory(source_path, adjust_path_prefix=False)
    split_metadata = BongardDatasetMetadata(
        images=[m for m in metadata.images if m.path in split_images]
    )
    split_metadata.to_file(Path(output_path, "metadata.json"))


def main():
    parser = argparse.ArgumentParser(
        description=("Split the dataset metadata to train and test splits.")
    )
    parser.add_argument(
        "--dataset",
        help="Path to the original dataset directory.",
    )
    parser.add_argument(
        "--output-train-split",
        help="Outout path to train split.",
    )
    parser.add_argument(
        "--output-test-split",
        help="Outout path to test split.",
    )
    parser.add_argument(
        "--test-size",
        help="Number of problems in the test split.",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--random-seed",
        help="Random seed for reproducibility.",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    dataset = args.dataset
    output_train_split = args.output_train_split
    output_test_split = args.output_test_split
    test_size = args.test_size
    random_seed = args.random_seed

    print(f"Dataset: {dataset}")
    print(f"Output train split: {output_train_split}")
    print(f"Output test split: {output_test_split}")
    print(f"Test size: {test_size}")
    print(f"Random seed: {random_seed}")

    np.random.seed(random_seed)

    dataset_info = BongardDatasetInfo.from_directory(dataset, adjust_path_prefix=False)
    train_problems, test_problems = train_test_split(dataset_info, test_size=test_size)

    copy_problems_by_concepts(
        dataset_info,
        train_problems,
        source_path=dataset,
        output_path=output_train_split,
    )
    copy_problems_by_concepts(
        dataset_info,
        test_problems,
        source_path=dataset,
        output_path=output_test_split,
    )


if __name__ == "__main__":
    main()

