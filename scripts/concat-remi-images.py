import argparse
import os
from pathlib import Path
from typing import Any, Dict

import datasets
from src.grid_renderers.remi import RemiGridRenderer


def concat_images(
    example: Dict[str, Any],
    padded_image_width: int,
    padded_image_height: int,
    font_path: str,
    font_size: int,
) -> Dict[str, Any]:
    renderer = RemiGridRenderer()

    image_cols = [col for col in example.keys() if col.startswith("image")]
    images = [example.pop(col) for col in image_cols]
    images = [img for img in images if img is not None]

    canvas, positions = renderer.draw_padded_remi_problem(
        images,
        padded_image_width=padded_image_width,
        padded_image_height=padded_image_height,
        font_path=font_path,
        font_size=font_size,
    )

    example["image"] = canvas
    example["positions"] = str(positions)

    return example


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate left and right images using RemiGridRenderer."
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
        default=12,
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
        lambda example: concat_images(
            example,
            args.padded_image_width,
            args.padded_image_height,
            args.font_path,
            args.font_size,
        ),
        batched=False,
        num_proc=args.num_shards,
        desc="Concatenating images",
        writer_batch_size=10,
        load_from_cache_file=False,
    )

    columns_to_drop = [
        col for col in processed_dataset.column_names if col.startswith("image_")
    ]

    if columns_to_drop:
        processed_dataset = processed_dataset.remove_columns(columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")

    processed_dataset.save_to_disk(args.output)
    print(f"Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
