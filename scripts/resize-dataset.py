import argparse
import os
from typing import Any, Dict

from PIL import Image

import datasets
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def resize_images(
    example: Dict[str, Any],
    patch_size: int,
    merge_size: int,
) -> Dict[str, Any]:
    image_columns = [col for col in example if col.startswith("image")]

    for col in image_columns:
        image_batch = example[col]
        resized_images = []
        for image_data in image_batch:
            image: Image.Image = image_data

            image_width, image_height = image.size
            desired_width, desired_height = smart_resize(
                image_width,
                image_height,
                patch_size * merge_size,
            )
            resized_image = image.resize((desired_width, desired_height))
            resized_images.append(resized_image)

        example[col] = resized_images

    return example


def main():
    parser = argparse.ArgumentParser(
        description="Resize Bongard data to fit the provided patch size."
    )
    parser.add_argument("--input", help="Path to the dataset directory.")
    parser.add_argument("--output", help="Path to the output directory.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of shards.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--merge-size",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    input = args.input
    output = args.output
    num_shards = args.num_shards
    patch_size = args.patch_size
    merge_size = args.merge_size

    if not os.path.isdir(input):
        parser.error(f"Input directory '{input}' does not exist or is not a directory.")

    print(f"Input directory: {input}")
    print(f"Output directory: {output}")
    print(f"Number of shards: {num_shards}")
    print(f"Patch size: {patch_size}")
    print(f"Merge size: {merge_size}")

    input_dataset = datasets.load_from_disk(input)
    resized_dataset = input_dataset.map(
        lambda example: resize_images(example, patch_size, merge_size),
        batched=True,
        num_proc=num_shards,
        writer_batch_size=10,
        batch_size=10,
    )

    os.makedirs(output, exist_ok=True)
    resized_dataset.save_to_disk(output)
    print(f"Dataset saved to: {output}")


if __name__ == "__main__":
    main()
