from pydantic import BaseModel
from typing import Union

from src.model.base import SerializableModel


class ImageAttributions(BaseModel):
    column_name: str
    entities: list[str]
    mask_files: list[list[str]]


class ProblemAttributions(BaseModel):
    problem_id: Union[int, str]
    images: list[ImageAttributions]


class AttributionResults(SerializableModel):
    attributions: list[ProblemAttributions]

    @classmethod
    def from_directory(
        cls,
        path: str,
        adjust_image_paths: bool = False,
        attributions_file: str = "attributions.json",
    ) -> "AttributionResults":
        attrs = cls.from_file(f"{path}/{attributions_file}")

        if adjust_image_paths:
            for problem_attr in attrs.attributions:
                for img_attr in problem_attr.images:
                    img_attr.mask_files = [
                        f"{path}/{relative_path}"
                        for relative_path in img_attr.mask_files
                    ]

        return attrs
