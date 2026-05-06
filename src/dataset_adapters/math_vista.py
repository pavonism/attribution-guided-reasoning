from typing import List, Union

from PIL.Image import Image

from src.dataset_adapters.base import ConversationFormat, DatasetAdapter
from src.dataset_adapters.common import make_image_content, make_text_content


class MathVistaAdapter(DatasetAdapter):
    def __init__(
        self,
        text_only: bool,
        conversation_format: ConversationFormat,
        format_prompt: str = "",
    ):
        super().__init__(text_only, conversation_format, format_prompt)

    def make_conversations(self, batch: dict) -> list[list[dict]]:
        questions = batch["query"]
        images = batch["decoded_image"]

        conversations = []
        for question, image in zip(questions, images):
            contents = []

            if self._format_prompt:
                question += "\n\n" + self._format_prompt

            contents.append(make_text_content(question))
            contents.append(make_image_content(image, self._conversation_format))
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": contents,
                    }
                ]
            )

        return conversations

    def get_problem_ids(self, batch: dict) -> list[Union[int, str]]:
        return batch["pid"]

    def get_questions(self, batch: dict) -> List[str]:
        return batch["query"]

    def get_answers(self, batch: dict) -> list[str]:
        expected_answers = []

        for index, answer in enumerate(batch["answer"]):
            question_type = batch["question_type"][index]
            if question_type == "multi_choice":
                choices = batch["choices"][index]
                epected_answer_index = choices.index(answer)
                expected_answer_letter = chr(ord("A") + epected_answer_index)
                expected_answers.append(expected_answer_letter)
            else:
                expected_answers.append(answer)

        return expected_answers

    def get_images(self, batch: dict) -> list[list[Image]]:
        return [[img] for img in batch["decoded_image"]]

    def get_image_columns(self) -> list[str]:
        return ["decoded_image"]
