from pydantic import BaseModel
from typing import Union
from src.model.base import SerializableModel


class ImageAttributions(SerializableModel):
    column_name: str
    entities: list[str]


class ProblemAttributions(BaseModel):
    problem_id: Union[str, int]
    images: list[ImageAttributions]


class EntityAttributionResults(SerializableModel):
    attributions: list[ProblemAttributions]
