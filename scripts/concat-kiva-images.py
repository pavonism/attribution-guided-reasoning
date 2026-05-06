import argparse
import os
from typing import Any, Dict

import datasets
from src.grid_renderers.kiva import KivaGridRenderer

NEW_PROMPT = """
You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using
visual analogical reasoning. You will think step-by-step and carefully examine the visual evidence
before providing an answer.

Observe the following left-to-right transformation of an object in the top row. The object picture on the left transforms to
the object picture on the right. Denote this transformation as training transformation.

Now you are given three new pictures. Each new picture contains a left-to-right transformation of a
new object (marked by either (A), (B) or (C)). Which one of these three left-to-right
transformations follows the original training transformation?
""".strip()


def concat_images(
    example: Dict[str, Any],
    padded_image_width: int,
    padded_image_height: int,
) -> Dict[str, Any]:
    renderer = KivaGridRenderer()

    image_cols = [col for col in example.keys() if col.startswith("image")]
    images = [example.pop(col) for col in image_cols]
    images = [img for img in images if img is not None]

    canvas, positions = renderer.draw_padded_kiva_problem(
        images,
        padded_image_width=padded_image_width,
        padded_image_height=padded_image_height,
    )

    example["question"] = NEW_PROMPT
    example["image"] = canvas
    example["positions"] = str(positions)

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
        lambda example: concat_images(
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
        col for col in processed_dataset.column_names if col.startswith("image_")
    ]

    if columns_to_drop:
        processed_dataset = processed_dataset.remove_columns(columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")

    processed_dataset.save_to_disk(args.output)
    print(f"Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
