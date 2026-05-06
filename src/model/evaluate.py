from typing import List, Union
from src.model.base import SerializableModel


class EvaluatedProblem(SerializableModel):
    id: Union[int, str]
    problem: str
    model_output: str
    extracted_answer: str
    extracted_thought: str
    solution: str
    correct: bool


class EvaluateResults(SerializableModel):
    evaluations: List[EvaluatedProblem]

    def to_dict(self) -> dict[int, EvaluatedProblem]:
        return {x.id: x for x in self.evaluations}
