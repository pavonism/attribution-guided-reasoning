from pydantic import BaseModel


class ExperimentResult(BaseModel):
    problem_id: int
    image_cols: list[str]
    prompt: str
    response: str


class ExperimentResults(BaseModel):
    results: list[ExperimentResult]
