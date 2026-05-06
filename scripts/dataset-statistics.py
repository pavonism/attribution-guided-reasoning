import argparse
import sys
from pathlib import Path
from typing import Set

from datasets import load_from_disk, Dataset
from tqdm import tqdm


def get_image_columns(dataset: Dataset) -> list[str]:
    """Extract all image column names from the dataset."""
    return [col for col in dataset.column_names if col.startswith("image")]


def count_unique_images(dataset: Dataset, image_columns: list[str]) -> int:
    seen_images: Set[str] = set()
    batch_size = 1000

    total_rows = len(dataset)

    for start_idx in tqdm(
        range(0, total_rows, batch_size),
        desc="Counting unique images",
    ):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataset.select(range(start_idx, end_idx))

        for row_idx in range(len(batch)):
            for col in image_columns:
                img = batch[col][row_idx]
                if img is not None:
                    img_key = None

                    if hasattr(img, "filename") and img.filename:
                        img_key = img.filename
                    elif hasattr(img, "src") and img.src:
                        img_key = img.src
                    else:
                        img_key = id(img)

                    if img_key is not None:
                        seen_images.add(img_key)

    return len(seen_images)


def count_unique_concepts(dataset: Dataset, solution_column: str = "solution") -> int:
    if solution_column not in dataset.column_names:
        print(f"Warning: '{solution_column}' column not found", file=sys.stderr)
        return 0

    unique_concepts: Set[str] = set()
    batch_size = 1000
    total_rows = len(dataset)

    for start_idx in tqdm(
        range(0, total_rows, batch_size),
        desc="Counting unique concepts",
    ):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataset.select(range(start_idx, end_idx))

        for concept in batch[solution_column]:
            if concept is not None:
                unique_concepts.add(concept)

    return len(unique_concepts)


def count_unique_entities(dataset: Dataset) -> int:
    entity_columns = [col for col in dataset.column_names if col.startswith("entities")]
    seen_entities: Set[str] = set()
    batch_size = 1000
    total_rows = len(dataset)

    for start_idx in tqdm(
        range(0, total_rows, batch_size),
        desc="Counting unique entities",
    ):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataset.select(range(start_idx, end_idx))

        for row_idx in range(len(batch)):
            for col in entity_columns:
                entities = batch[col][row_idx]
                if entities is not None:
                    seen_entities.update(entities)

    return len(seen_entities)


def count_masks(dataset: Dataset) -> int:
    mask_columns = [col for col in dataset.column_names if col.startswith("masks")]
    total_masks = 0
    batch_size = 1000
    total_rows = len(dataset)

    for start_idx in tqdm(
        range(0, total_rows, batch_size),
        desc="Counting masks",
    ):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataset.select(range(start_idx, end_idx))

        for row_idx in range(len(batch)):
            for col in mask_columns:
                entity_masks = batch[col][row_idx]
                if entity_masks is not None:
                    for masks in entity_masks:
                        if masks is not None:
                            total_masks += len(masks)

    return total_masks


def main():
    parser = argparse.ArgumentParser(description="Compute Bongard dataset statistics.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset on local disk",
    )
    parser.add_argument(
        "--solution_column",
        type=str,
        default="solution",
        help="Name of the column containing the solution/concept",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")

    dataset = load_from_disk(dataset_path)

    print(f"\n{'=' * 60}")
    print("Dataset Statistics")
    print(f"{'=' * 60}")

    total_rows = len(dataset)
    print(f"Total number of rows: {total_rows}")

    print("\nCounting unique concepts...")
    unique_concepts = count_unique_concepts(
        dataset,
        solution_column=args.solution_column,
    )
    print(f"Number of unique concepts: {unique_concepts}")

    print("\nCounting unique entities...")
    unique_entities = count_unique_entities(dataset)
    print(f"Number of unique entities: {unique_entities}")

    print("\nCounting masks...")
    total_masks = count_masks(dataset)
    print(f"Total number of masks: {total_masks}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
