from typing import List, Union
from PIL.Image import Image
from enum import Enum


class ConversationFormat(Enum):
    TRANSFORMERS = "transformers"
    VLLM = "vllm"


class DatasetAdapter:
    def __init__(
        self,
        text_only: bool,
        conversation_format: ConversationFormat,
        format_prompt: str = "",
    ):
        self._text_only = text_only
        self._conversation_format = conversation_format
        self._format_prompt = format_prompt

    def make_conversations(self, batch: dict) -> list[list[dict]]:
        pass

    def get_problem_ids(self, batch: dict) -> list[Union[int, str]]:
        pass

    def get_questions(self, batch: dict) -> List[str]:
        pass

    def get_answers(self, batch: dict) -> list[str]:
        pass

    def get_images(self, batch: dict) -> list[list[Image]]:
        pass

    def get_image_columns(self) -> list[str]:
        pass
