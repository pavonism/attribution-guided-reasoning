import argparse
from pathlib import Path
import sys
from datasets import load_from_disk, Dataset
from PIL import Image
from typing import Optional

from src.masks import prepare_mask, smart_resize, load_mask_from_bytes


def export_mask(
    dataset: Dataset,
    problem_index: int,
    problem_id: Optional[int],
    image_column: str,
    output_path: str,
    patch_size: int,
    max_pixels: int,
    min_pixels: int,
    all_masks: bool = False,
    keep_size: bool = False,
):
    print("Finding problem with ID:", problem_id)

    if problem_id is not None:
        if "id" not in dataset.column_names:
            print(
                f"Error: No id column in the dataset."
            )
            return
        else:
            problems_ids = dataset["id"]
            problem_index = problems_ids.index(problem_id)

    problem_row = dataset[problem_index]

    image: Image.Image = problem_row[image_column]

    if not keep_size:
        print("Resizing image...")
        w, h = image.size
        h_bar, w_bar = smart_resize(h, w, patch_size, min_pixels, max_pixels)
        image = image.resize((w_bar, h_bar), resample=Image.BILINEAR)

    print(f"Saving image to: {output_path}")
    image.save(output_path)

    masks_column = image_column.replace("image", "masks")
    if masks_column not in dataset.column_names:
        print(
            f"Error: No corresponding masks column found for image column '{image_column}'"
        )
        return

    mask_data = problem_row[masks_column]

    entities_column = image_column.replace("image", "entities")
    entities = problem_row[entities_column]

    print(f"Entities: {entities}")

    if mask_data is None or len(mask_data) == 0:
        print(
            f"Error: No mask found in column '{masks_column}' for problem ID {problem_id}"
        )
        return

    if all_masks:
        print("Exporting all masks for each entity...")

        for entity, entity_masks in zip(entities, mask_data):
            print(f"Entity: {entity}, Number of masks: {len(entity_masks)}")
            entity_name = entity.replace(" ", "-").lower()
            for index, mask_dict in enumerate(entity_masks):
                output_file = Path(output_path).with_stem(
                    Path(output_path).stem + f"-{entity_name}-{index}"
                )
                mask = load_mask_from_bytes(mask_dict["bytes"])
                mask_image = Image.fromarray(mask.cpu().numpy().astype("uint8") * 255)
                print(f"Saving mask for entity '{entity}' to: {output_file}")
                mask_image.save(output_file)

    try:
        print("Preparing mask...")
        mask = prepare_mask(mask_data, h_bar, w_bar)
        mask_image = Image.fromarray(mask.cpu().numpy().astype("uint8") * 255)
        mask_output_path = Path(output_path).with_stem(Path(output_path).stem + "-mask")
        print(f"Saving mask to: {mask_output_path}")
        mask_image.save(mask_output_path)
    except Exception as e:
        print(f"Error: Failed to export mask - {e}")


def main():
    parser = argparse.ArgumentParser(description="Compute Bongard dataset statistics.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset on local disk",
    )
    parser.add_argument(
        "--problem-id",
        type=int,
        help="ID of the problem to export the mask for.",
    )
    parser.add_argument(
        "--problem-index",
        type=int,
        help="Index of the problem to export the mask for.",
        default=0,
    )
    parser.add_argument(
        "--image-column",
        type=str,
        help="Column name containing the image to export.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        help="Size of the patch to adjust the image to.",
        default=28,
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Maximum number of pixels in the output image.",
        default=401408,
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        help="Minimum number of pixels in the output image.",
        default=3136,
    )
    parser.add_argument(
        "--all-masks",
        action="store_true",
        help="Export all masks.",
    )
    parser.add_argument(
        "--keep-size",
        action="store_true",
        help="Keep original image size without resizing.",
    )

    args = parser.parse_args()
    print(args)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")

    image_columns = (
        [args.image_column]
        if args.image_column
        else [col for col in dataset.column_names if col.startswith("image")]
    )

    output_paths = [
        f"{args.output}/bmerged-{args.problem_id}-{col.replace('image_', '').replace('_', '-')}.jpg"
        for col in image_columns
    ]

    for image_column, output_path in zip(image_columns, output_paths):
        export_mask(
            dataset,
            problem_index=args.problem_index,
            problem_id=args.problem_id,
            image_column=image_column,
            output_path=output_path,
            patch_size=args.patch_size,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            all_masks=args.all_masks,
            keep_size=args.keep_size,
        )


if __name__ == "__main__":
    main()
