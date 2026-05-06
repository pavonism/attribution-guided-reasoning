import argparse
import glob
import json
import os

from pathlib import Path
import shutil

from tqdm import tqdm
import numpy as np


from src.model.bongard import (
    BongardImageMetadata,
    BongardProblem,
    BongardDatasetInfo,
    BongardDatasetMetadata,
)


def load_hoi(paths: list[str], max_instances_per_concept: int) -> list:
    dataset = []

    for file in paths:
        with open(file, "r") as f:
            dataset.extend(json.load(f))

    concept_to_instances: dict[str, list] = {}
    for problem in dataset:
        concept = problem[2]
        if concept not in concept_to_instances:
            concept_to_instances[concept] = []
        concept_to_instances[concept].append(problem)

    # Limit the number of instances per concept
    limited_dataset = []
    for concept, instances in concept_to_instances.items():
        sampled_indexes = np.random.choice(
            len(instances),
            min(len(instances), max_instances_per_concept),
            replace=False,
        )
        limited_dataset.extend([instances[i] for i in sampled_indexes])

    return limited_dataset


def to_bongard_dataset(
    hoi: list,
    input_dir: str,
    output_dir: str,
) -> BongardDatasetInfo:
    bongard_dataset = BongardDatasetInfo(problems=[])
    bongard_metadata = BongardDatasetMetadata(images=[])
    image_index = 0

    for problem_id, problem in enumerate(tqdm(hoi, desc="Loading problems...")):
        left_images = []
        right_images = []

        concept = problem[2]

        left_concept = concept.replace("++", " ").replace("_", " ")
        right_concept = "no " + left_concept

        for index, side in enumerate(["left", "right"]):
            for img_data in problem[index]:
                img_path = img_data["im_path"]
                ext = Path(img_path).suffix
                new_img_path = f"images/{image_index:05d}{ext}"
                bongard_metadata.images.append(
                    BongardImageMetadata(
                        path=new_img_path,
                        side=side,
                        problem_id=problem_id,
                        group_id=concept,
                    )
                )
                image_index += 1

                source_path = Path(input_dir, img_path)
                dest_path = Path(output_dir, new_img_path)
                os.makedirs(dest_path.parent, exist_ok=True)
                shutil.copy(source_path, dest_path)

                if side == "left":
                    left_images.append(new_img_path)
                else:
                    right_images.append(new_img_path)

        bongard_dataset.problems.append(
            BongardProblem(
                id=problem_id,
                left_concept=left_concept,
                right_concept=right_concept,
                left_images=left_images,
                right_images=right_images,
                concept_group=concept,
                dataset="bongard-hoi",
            )
        )

    bongard_dataset.to_file(Path(output_dir, "dataset.json"))
    bongard_metadata.to_file(Path(output_dir, "metadata.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert Bongard-OpenWorld dataset to Bongard-RWR format."
    )
    parser.add_argument("--input-dir", help="Path to the labels file.")
    parser.add_argument("--output-dir", help="Path to the output directory.")
    parser.add_argument(
        "--max-instances-per-concept",
        type=int,
        default=100,
        help="Number of instances per concept.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_instances_per_concept = args.max_instances_per_concept

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max instances per concept: {max_instances_per_concept}")

    np.random.seed(args.seed)
    os.makedirs(output_dir, exist_ok=True)

    train = load_hoi(
        [
            *glob.glob(f"{input_dir}/bongard_hoi_train.json"),
            *glob.glob(f"{input_dir}/bongard_hoi_val_*.json"),
            *glob.glob(f"{input_dir}/bongard_hoi_test_seen_obj_seen_act.json"),
        ],
        max_instances_per_concept,
    )

    test = load_hoi(
        glob.glob(f"{input_dir}/bongard_hoi_test_unseen_obj_unseen_act.json"),
        max_instances_per_concept,
    )

    to_bongard_dataset(train, input_dir, f"{output_dir}-train")
    to_bongard_dataset(test, input_dir, f"{output_dir}-test")

    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
