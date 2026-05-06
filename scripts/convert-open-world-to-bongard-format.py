import argparse
import os

from pathlib import Path
import shutil

from pydantic import BaseModel, RootModel, Field
from tqdm import tqdm


from src.model.bongard import (
    BongardImageMetadata,
    BongardProblem,
    BongardDatasetInfo,
    BongardDatasetMetadata,
)


class OpenWorldProblem(BaseModel):
    uid: str
    commonSense: str
    concept: str
    caption: str
    imageFiles: list[str]
    urls: list[str]


CONCEPT_GROUPS = {
    0: "other",
    1: "human-object-interaction",
    2: "taste-nutrition-food",
    3: "color-material-shape",
    4: "functionality-status-affordance",
    5: "and-or-not",
    6: "factual-knowledge",
    7: "meta-class",
    8: "relationship",
    9: "unusual-observations",
}


class OpenWorldDataset(RootModel):
    root: list[OpenWorldProblem] = Field(default_factory=list)

    def __getitem__(self, index: int) -> OpenWorldProblem:
        return self.root[index]

    def __iter__(self):
        return iter(self.root)

    def union(self, other: "OpenWorldDataset"):
        self.root = self.root + other.root


def load_open_world(input_file: str) -> OpenWorldDataset:
    dataset = OpenWorldDataset()
    with open(input_file, "r") as f:
        for line in f:
            problem = OpenWorldProblem.model_validate_json(line)
            dataset.root.append(problem)

    return dataset


def to_bongard_dataset(
    open_world: OpenWorldDataset,
    input_dir: str,
    output_dir: str,
) -> BongardDatasetInfo:
    bongard_dataset = BongardDatasetInfo(problems=[])
    bongard_metadata = BongardDatasetMetadata(images=[])
    image_index = 0

    for problem in tqdm(open_world, desc="Loading problems..."):
        left_images = []
        right_images = []

        for img_path, url in zip(problem.imageFiles, problem.urls):
            side = "left" if Path(img_path).name.startswith("pos") else "right"
            ext = Path(img_path).suffix
            new_img_path = f"images/{image_index:05d}{ext}"
            bongard_metadata.images.append(
                BongardImageMetadata(
                    path=new_img_path,
                    group_id=problem.uid,
                    caption=problem.caption,
                    url=url,
                    side=side,
                    problem_id=problem.uid,
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
                id=problem.uid,
                left_concept=problem.concept,
                right_concept="no " + problem.concept,
                left_images=left_images,
                right_images=right_images,
                concept_group=CONCEPT_GROUPS.get(int(problem.commonSense), "other"),
                dataset="bongard-open-world",
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

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    train = load_open_world(f"{input_dir}/train.jsonl")
    val = load_open_world(f"{input_dir}/val.jsonl")
    test = load_open_world(f"{input_dir}/test.jsonl")

    train.union(val)

    to_bongard_dataset(train, input_dir, f"{output_dir}-train")
    to_bongard_dataset(test, input_dir, f"{output_dir}-test")

    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
