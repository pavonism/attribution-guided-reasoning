import argparse
import os
import random

from pathlib import Path
import shutil

from tqdm import tqdm


from src.model.bongard import (
    BongardImageMetadata,
    BongardProblem,
    BongardDatasetInfo,
    BongardDatasetMetadata,
)


def merge_datasets(
    datasets: list[BongardDatasetInfo],
    metadatas: list[BongardDatasetMetadata],
    dataset_names: list[str],
    output_dir: str,
) -> BongardDatasetInfo:
    for dataset, name in zip(datasets, dataset_names):
        for problem in dataset.problems:
            problem.dataset = name

    merged_metadata = BongardDatasetMetadata(
        images=[image for md in metadatas for image in md.images]
    )
    merged_dataset = BongardDatasetInfo(
        problems=[problem for ds in datasets for problem in ds.problems]
    )
    random_order = list(range(len(merged_dataset.problems)))
    random.shuffle(random_order)

    metadata_dict = merged_metadata.to_dict()
    new_dataset = BongardDatasetInfo(problems=[])
    new_metadata = BongardDatasetMetadata(images=[])
    image_index = 0

    for index in tqdm(random_order, desc="Merging datasets..."):
        problem = merged_dataset.problems[index]

        new_problem_id = len(new_dataset.problems)
        new_left_images = []
        new_right_images = []

        for images, new_images in [
            (problem.right_images, new_right_images),
            (problem.left_images, new_left_images),
        ]:
            for img in images:
                img_metadata = metadata_dict.get(img)
                if img_metadata is None:
                    raise ValueError(f"Image metadata not found for image: {img}")

                suffix = Path(img_metadata.path).suffix
                new_img_path = f"images/{image_index:05d}{suffix}"
                shutil.copyfile(
                    img_metadata.path,
                    f"{output_dir}/{new_img_path}",
                )

                new_metadata.images.append(
                    BongardImageMetadata(
                        problem_id=new_problem_id,
                        side=img_metadata.side,
                        path=new_img_path,
                        group_id=img_metadata.group_id,
                        caption=img_metadata.caption,
                        url=img_metadata.url,
                    )
                )

                new_images.append(new_img_path)
                image_index += 1

        new_problem = BongardProblem(
            id=new_problem_id,
            left_images=new_left_images,
            right_images=new_right_images,
            left_concept=problem.left_concept,
            right_concept=problem.right_concept,
            concept_group=problem.concept_group,
            diversity=problem.diversity,
            dataset=problem.dataset,
            dataset_problem_id=problem.id,
        )

        new_dataset.problems.append(new_problem)

    new_dataset.to_file(Path(output_dir, "dataset.json"))
    new_metadata.to_file(Path(output_dir, "metadata.json"))

def main():
    parser = argparse.ArgumentParser(description="Merge Bongard datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Paths to Bongard datasets.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        help="Names of the Bongard datasets.",
    )
    parser.add_argument("--output-dir", help="Path to the output directory.")

    args = parser.parse_args()

    datasets = args.datasets
    dataset_names = args.dataset_names
    output_dir = args.output_dir

    print(f"Datasets: {datasets}")
    print(f"Dataset names: {dataset_names}")
    print(f"Output directory: {output_dir}")

    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    dataset_infos = [BongardDatasetInfo.from_directory(dataset) for dataset in datasets]
    metadata_infos = [
        BongardDatasetMetadata.from_directory(dataset) for dataset in datasets
    ]

    merge_datasets(dataset_infos, metadata_infos, dataset_names, output_dir)

    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
