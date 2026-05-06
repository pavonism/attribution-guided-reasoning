from typing import Dict, List, Optional

from pydantic import BaseModel
from src.model.base import SerializableModel


class BongardImageMetadata(BaseModel):
    problem_id: int
    side: str
    path: str
    group_id: Optional[str] = None
    caption: Optional[str] = None
    url: Optional[str] = None


class BongardDatasetMetadata(SerializableModel):
    images: List[BongardImageMetadata]

    def to_dict(self) -> Dict[str, BongardImageMetadata]:
        return {img.path: img for img in self.images}

    @staticmethod
    def from_directory(
        path: str,
        adjust_path_prefix: bool = True,
        metadata_file: str = "metadata.json",
    ) -> "BongardDatasetMetadata":
        with open(f"{path}/{metadata_file}", "r", encoding="utf-8") as f:
            json_str = f.read()
            metadata = BongardDatasetMetadata.model_validate_json(json_str)

            if adjust_path_prefix:
                for image in metadata.images:
                    image.path = f"{path}/{image.path}"

            return metadata


class BongardProblem(BaseModel):
    id: int
    left_images: List[str]
    right_images: List[str]
    left_side_image: str = ""
    right_side_image: str = ""
    whole_image: str = ""
    left_concept: str = ""
    right_concept: str = ""
    concept_group: str = ""
    diversity: Optional[float] = None
    dataset: Optional[str] = None
    dataset_problem_id: Optional[int] = None


class BongardDatasetInfo(BaseModel):
    problems: List[BongardProblem]

    def subset(self, problem_ids: List[int]) -> "BongardDatasetInfo":
        problems = [problem for problem in self.problems if problem.id in problem_ids]
        return BongardDatasetInfo(problems=problems)

    def collect_images(self) -> List[str]:
        images = []
        for problem in self.problems:
            for img in problem.left_images + problem.right_images:
                if img not in images:
                    images.append(img)
        return images

    @staticmethod
    def from_directory(
        path: str,
        adjust_path_prefix: bool = True,
        dataset_file: str = "dataset.json",
    ) -> "BongardDatasetInfo":
        with open(f"{path}/{dataset_file}", "r", encoding="utf-8") as f:
            json_str = f.read()
            dataset_info = BongardDatasetInfo.model_validate_json(json_str)

            if adjust_path_prefix:
                for problem in dataset_info.problems:
                    problem.left_images = [
                        f"{path}/{img}" for img in problem.left_images
                    ]
                    problem.right_images = [
                        f"{path}/{img}" for img in problem.right_images
                    ]
                    problem.left_side_image = f"{path}/{problem.left_side_image}"
                    problem.right_side_image = f"{path}/{problem.right_side_image}"
                    problem.whole_image = f"{path}/{problem.whole_image}"

            return dataset_info

    def to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json_str = self.model_dump_json(indent=4)
            f.write(json_str)
