import argparse
from datasets import Dataset, load_from_disk, load_dataset
from pathlib import Path
from PIL import Image
import numpy as np
import tqdm

from src.dataset_adapters import AutoDatasetAdapter
from src.dataset_adapters.base import ConversationFormat, DatasetAdapter
from src.model.mask_attribution import AttributionResults, ImageAttributions


def overlay_masks(image: Image.Image, masks: np.ndarray) -> Image.Image:
    image = image.convert("RGBA")
    masks = 100 * masks.astype(np.uint8)

    for mask in masks:
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, (0, 255, 0, 0))
        overlay.putalpha(mask)
        image = Image.alpha_composite(image, overlay)
    return image


def visualize_image_attributions(
    image: Image.Image,
    attributions: ImageAttributions,
    output_path: Path,
):
    masks = np.array(attributions.masks)
    final_image = overlay_masks(image, masks)
    final_image.save(output_path)


def visualize_attributions(
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    attributions_file: str,
    output_path: str,
):
    attributions = AttributionResults.from_file(attributions_file)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pid_to_index = {
        pid: idx for idx, pid in enumerate(dataset_adapter.get_problem_ids(dataset))
    }

    for problem_attr in tqdm.tqdm(attributions.attributions[:10]):
        problem_dir = output_dir.joinpath(f"{problem_attr.problem_id}")

        if problem_dir.exists():
            continue

        problem_dir.mkdir(parents=True, exist_ok=True)
        problem_index = pid_to_index[problem_attr.problem_id]
        row = dataset[problem_index]

        for image_attr in problem_attr.images:
            image = row[image_attr.column_name]
            image_output_path = problem_dir.joinpath(
                f"{image_attr.column_name}_attributions.png"
            )

            visualize_image_attributions(
                image,
                image_attr,
                image_output_path,
            )


def main():
    parser = argparse.ArgumentParser(description="Visualize mask attributions.")
    parser.add_argument("--dataset", help="Path to test dataset")
    parser.add_argument("--attributions", help="Path to a file with attributions.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the attribution results.",
    )

    args = parser.parse_args()
    dataset_name_or_path = args.dataset
    attributions = args.attributions
    output_path = args.output_dir

    print(f"Dataset: {dataset_name_or_path}")
    print(f"Attributions file: {attributions}")
    print(f"Output directory: {output_path}")

    print("Loading dataset...")
    dataset = (
        load_from_disk(dataset_name_or_path)
        if Path(dataset_name_or_path).exists()
        else load_dataset(dataset_name_or_path)
    )
    dataset_adapter = AutoDatasetAdapter.from_dataset(
        dataset_name_or_path,
        dataset,
        text_only=False,
        conversation_format=ConversationFormat.TRANSFORMERS,
    )

    print("Running experiment...")
    visualize_attributions(
        dataset,
        dataset_adapter,
        attributions,
        output_path,
    )


if __name__ == "__main__":
    main()
