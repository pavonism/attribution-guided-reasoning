import argparse
import os
from typing import Any, Dict

from PIL import Image
import datasets
import torch
from src.grid_renderers.bongard import BongardGridRenderer
from src.masks import load_mask_from_bytes


def concat_images(
    example: Dict[str, Any],
    padded_image_width: int,
    padded_image_height: int,
) -> Dict[str, Any]:
    renderer = BongardGridRenderer()

    left_images = [example.pop(f"image_left_{i}") for i in range(6)]
    right_images = [example.pop(f"image_right_{i}") for i in range(6)]

    canvas, left_positions, right_positions = renderer.draw_padded_bongard_problem(
        tuple(left_images),
        tuple(right_images),
        padded_image_width=padded_image_width,
        padded_image_height=padded_image_height,
    )

    example["image"] = canvas
    example["positions_left"] = str(left_positions)
    example["positions_right"] = str(right_positions)

    return example


def union_masks(
    entity_masks: list[list[dict]],
    image: Image.Image,
) -> list[Image.Image]:
    masks = [
        load_mask_from_bytes(mask_dict["bytes"])
        for mask_dicts in entity_masks
        for mask_dict in mask_dicts
    ]

    masks = [
        mask
        for mask in masks
        if mask.shape[0] == image.size[1] and mask.shape[1] == image.size[0]
    ]

    if len(masks) == 0:
        return Image.new("L", image.size, color=0)

    all_masks = torch.stack(masks, dim=0)
    union_mask = torch.any(all_masks, dim=0)
    union_mask_image = Image.fromarray(union_mask.cpu().numpy().astype("uint8") * 255)
    return union_mask_image


def concat_masks(
    example: Dict[str, Any],
    padded_image_width: int,
    padded_image_height: int,
) -> Dict[str, Any]:
    renderer = BongardGridRenderer()

    left_images = [example.get(f"image_left_{i}") for i in range(6)]
    right_images = [example.get(f"image_right_{i}") for i in range(6)]

    left_masks = [example.pop(f"masks_left_{i}") for i in range(6)]
    right_masks = [example.pop(f"masks_right_{i}") for i in range(6)]

    left_masks = [
        union_masks(mask_dicts, image)
        for mask_dicts, image in zip(left_masks, left_images)
    ]
    right_masks = [
        union_masks(mask_dicts, image)
        for mask_dicts, image in zip(right_masks, right_images)
    ]

    canvas, *_ = renderer.draw_padded_bongard_problem(
        tuple(left_masks),
        tuple(right_masks),
        padded_image_width=padded_image_width,
        padded_image_height=padded_image_height,
        padded_color="black",
        separator_color="black",
    )

    example["mask"] = canvas
    return example


def map(
    example: Dict[str, Any],
    padded_image_width: int,
    padded_image_height: int,
) -> Dict[str, Any]:
    if any(col.startswith("masks_") for col in example.keys()):
        example = concat_masks(example, padded_image_width, padded_image_height)

    example = concat_images(example, padded_image_width, padded_image_height)
    return example


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate left and right images using BongardGridRenderer."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the input Hugging Face dataset directory.",
    )
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument(
        "--padded-image-width",
        type=int,
        default=512,
        help="Output image width in pixels.",
    )
    parser.add_argument(
        "--padded-image-height",
        type=int,
        default=512,
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=None,
        help="Maximum number of problems to process (useful for testing).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of shards for parallel processing.",
    )

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.dataset):
        parser.error(
            f"Input directory '{args.dataset}' does not exist or is not a directory."
        )

    if args.n_problems:
        print(f"Max problems to process: {args.n_problems}")
    print(f"Number of shards: {args.num_shards}")

    os.makedirs(args.output, exist_ok=True)

    input_dataset = datasets.load_from_disk(args.dataset)

    if args.n_problems:
        input_dataset = input_dataset.take(args.n_problems)

    processed_dataset = input_dataset.map(
        lambda example: map(
            example,
            args.padded_image_width,
            args.padded_image_height,
        ),
        batched=False,
        num_proc=args.num_shards,
        desc="Concatenating images",
        writer_batch_size=10,
        load_from_cache_file=False,
    )

    columns_to_drop = [
        col
        for col in processed_dataset.column_names
        if col.startswith("image_left_")
        or col.startswith("image_right_")
        or col.startswith("mask_left_")
        or col.startswith("mask_right_")
    ]

    if columns_to_drop:
        processed_dataset = processed_dataset.remove_columns(columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")

    processed_dataset.save_to_disk(args.output)
    print(f"Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
