import argparse
import os
from pathlib import Path
from typing import Any, Dict

import datasets

from src.grid_renderers.remi import RemiGridRenderer
from src.masks import adjust_to_patch_size


def resize_and_split_images(
    example: Dict[str, Any],
    padded_image_width: int,
    padded_image_height: int,
    patch_size: int,
    merge_size: int,
    max_pixels: int,
    font_path: str,
    font_size: int,
    margin: int,
) -> Dict[str, Any]:
    renderer = RemiGridRenderer()

    image_cols = sorted([col for col in example.keys() if col.startswith("image")])
    images = [example.get(col) for col in image_cols]
    images = [img for img in images if img is not None]

    canvas, _ = renderer.draw_padded_remi_problem(
        images,
        padded_image_width=padded_image_width,
        padded_image_height=padded_image_height,
        font_path=font_path,
        font_size=font_size,
        margin=margin,
    )

    resized_canvas = adjust_to_patch_size(
        canvas,
        patch_size=patch_size,
        merge_size=merge_size,
        max_pixels=max_pixels,
    )

    n_images = len(images)
    grid_positions = renderer.get_grid_positions(
        n_images,
        padded_image_width,
        padded_image_height,
        font_size,
        margin,
    )

    original_width, original_height = canvas.size
    new_width, new_height = resized_canvas.size
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    for idx, col_name in enumerate(image_cols[:n_images]):
        x_start_orig, y_start_orig, x_end_orig, y_end_orig = grid_positions[idx]

        x_start = int(x_start_orig * scale_x)
        y_start = int(y_start_orig * scale_y)
        x_end = int(x_end_orig * scale_x)
        y_end = int(y_end_orig * scale_y)

        cropped_image = resized_canvas.crop((x_start, y_start, x_end, y_end))
        example[col_name] = cropped_image

    return example


def main():
    parser = argparse.ArgumentParser(
        description="Resize concatenated Remi grid and split back into individual image columns."
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
        help="Output image width in pixels for each padded image.",
    )
    parser.add_argument(
        "--padded-image-height",
        type=int,
        default=512,
        help="Output image height in pixels for each padded image.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Patch size for image adjustment.",
    )
    parser.add_argument(
        "--merge-size",
        type=int,
        default=2,
        help="Merge size for image adjustment.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1568000,
        help="Maximum number of pixels for resized image.",
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
    parser.add_argument(
        "--split",
        help="Dataset split to evaluate on, if the dataset has splits.",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--font-path",
        help="Path to font file.",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--font-size",
        help="Font size for image tags.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--margin",
        help="Margin between grid cells in pixels.",
        type=int,
        default=15,
    )

    args = parser.parse_args()
    print(args)

    if args.n_problems:
        print(f"Max problems to process: {args.n_problems}")
    print(f"Number of shards: {args.num_shards}")

    os.makedirs(args.output, exist_ok=True)

    input_dataset = (
        datasets.load_from_disk(args.dataset)
        if Path(args.dataset).exists()
        else datasets.load_dataset(args.dataset, split=args.split)
    )

    if args.n_problems:
        input_dataset = input_dataset.take(args.n_problems)

    processed_dataset = input_dataset.map(
        lambda example: resize_and_split_images(
            example,
            args.padded_image_width,
            args.padded_image_height,
            args.patch_size,
            args.merge_size,
            args.max_pixels,
            args.font_path,
            args.font_size,
            args.margin,
        ),
        batched=False,
        num_proc=args.num_shards,
        desc="Resizing and splitting images",
        writer_batch_size=10,
        load_from_cache_file=False,
    )

    processed_dataset.save_to_disk(args.output)
    print(f"Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
