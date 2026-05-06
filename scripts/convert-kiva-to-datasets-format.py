import argparse
import os
from typing import Dict

from PIL import Image
from pathlib import Path
import glob

from datasets import Dataset
from pydantic import BaseModel, RootModel, Field
from tqdm import tqdm


PROMPT = """
You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using
visual analogical reasoning. You will think step-by-step and carefully examine the visual evidence
before providing an answer.

Observe the following left-to-right transformation of an object. The object picture on the left transforms to
the object picture on the right. Denote this transformation as training transformation.

<image1>

Now you are given three new pictures. Each new picture contains a left-to-right transformation of a
new object (marked by either (A), (B) or (C)). Which one of these three left-to-right
transformations follows the original training transformation?

<image2> <image3> <image4>
""".strip()


class KivaProblem(BaseModel):
    transform: str
    correct: str
    incorrect: str
    nochange: str
    train_input_value: str
    train_output_value: str
    test_input_value: str
    incorrect_test_output_value: str


class KivaDataset(RootModel):
    root: Dict[str, KivaProblem] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> KivaProblem:
        return self.root[key]

    def items(self):
        return self.root.items()

    def union(self, other: "KivaDataset"):
        self.root = self.root | other.root


def load_kiva(input_dir: str) -> KivaDataset:
    dataset = KivaDataset()
    for file in glob.glob(f"{input_dir}/test/*.json"):
        with open(file, "r") as f:
            content = f.read()
            current_dataset = KivaDataset.model_validate_json(content)
            dataset.union(current_dataset)

    return dataset


def to_datasets_dataset(kiva: KivaDataset, input_dir: str) -> Dataset:
    data = []

    for img_id, problem in tqdm(kiva.items(), desc="Loading problems..."):
        data_row = {}

        parts = img_id.split("_")
        transform, variation, regeneration = (
            parts[0],
            parts[1],
            parts[2] if len(parts) > 2 else None,
        )
        train_id = "_".join(parts[:2])

        data_row["id"] = img_id
        data_row["question"] = PROMPT
        data_row["correct"] = problem.correct
        data_row["image_1"] = Image.open(
            Path(input_dir, f"{train_id}_train.jpg")
        ).copy()
        data_row["image_2"] = Image.open(
            Path(input_dir, f"{transform}_{variation}_{regeneration}_test_0.jpg")
        ).copy()
        data_row["image_3"] = Image.open(
            Path(input_dir, f"{transform}_{variation}_{regeneration}_test_1.jpg")
        ).copy()
        data_row["image_4"] = Image.open(
            Path(input_dir, f"{transform}_{variation}_{regeneration}_test_2.jpg")
        ).copy()

        data.append(data_row)

    dataset = Dataset.from_list(data)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Bongard-RWR VIRFT data.")
    parser.add_argument("--input_dir", help="Path to the labels file.")
    parser.add_argument("--output_dir", help="Path to the output directory.")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not Path(input_dir, "test").exists():
        parser.error(
            "Input directory doesn't contain required 'test' folder with JSON problem definitions."
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    kiva_dataset = load_kiva(input_dir)
    dataset = to_datasets_dataset(kiva_dataset, input_dir)

    dataset.save_to_disk(output_dir)
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
