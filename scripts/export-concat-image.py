import argparse
import ast
import sys
from pathlib import Path
from typing import Optional
import numpy as np

import datasets
from PIL import Image
import json

from src.grid_renderers.bongard import BongardGridRenderer


def export_concat_image(
    dataset: datasets.Dataset,
    problem_id: Optional[int],
    column_name: str,
    output_path: str,
    drop_images: Optional[int] = None,
):
    print(f"Finding problem with ID: {problem_id}")
    problem_row = dataset[0]

    if "id" in dataset.features:
        problem_ids = dataset["id"]

        if problem_id and problem_id in problem_ids:
            problem_index = problem_ids.index(problem_id)
            problem_row = dataset[problem_index]

    if column_name not in problem_row or problem_row[column_name] is None:
        print(f"Error: No {column_name} found for problem ID {problem_id}")
        return False

    image: Image.Image = problem_row[column_name]

    if drop_images is not None and drop_images > 0:
        left_poses = problem_row.get("positions_left")
        right_poses = problem_row.get("positions_right")

        renderer = BongardGridRenderer()
        image = renderer.drop_images(
            image,
            np.array(ast.literal_eval(left_poses))[
                np.random.choice(range(6), size=drop_images, replace=False)
            ],
            np.array(ast.literal_eval(right_poses))[
                np.random.choice(range(6), size=drop_images, replace=False)
            ],
        )

    output_file = Path(output_path) / f"bmerged-{column_name}-concat-{problem_id}.jpg"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving image to: {output_file}")
    image.save(output_file)

    if "positions_left" in problem_row and problem_row["positions_left"] is not None:
        positions_file = output_file.with_stem(
            output_file.stem + "-positions"
        ).with_suffix(".json")

        positions_dict = {
            "positions_left": problem_row["positions_left"],
            "positions_right": problem_row["positions_right"],
        }

        with open(positions_file, "w") as f:
            json.dump(positions_dict, f, indent=4)

        print(f"Saved position metadata to: {positions_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export concatenated image for a specific problem ID."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset on local disk.",
    )
    parser.add_argument(
        "--problem-id",
        required=False,
        help="ID of the problem to export.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Export mask images instead of concatenated images.",
    )
    parser.add_argument(
        "--drop-images",
        type=int,
        help="Number of images to drop from the left and right of the concatenated image. Should be a non-negative integer.",
    )

    args = parser.parse_args()
    print(args)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")
    dataset = datasets.load_from_disk(dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")

    column_name = "mask" if args.mask else "image"
    success = export_concat_image(
        dataset,
        args.problem_id,
        column_name,
        args.output,
        args.drop_images,
    )

    if not success:
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
