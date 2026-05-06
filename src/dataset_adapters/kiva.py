from PIL.Image import Image

from datasets import Dataset

from src.dataset_adapters.base import DatasetAdapter, ConversationFormat
from src.dataset_adapters.common import (
    make_image_conversation,
    make_interleaved_image_conversation,
    make_multi_image_conversation,
    get_images_using_tags,
)


class KivaAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        text_only: bool,
        conversation_format: ConversationFormat,
        format_prompt: str = "",
        interleave_images: bool = True,
    ):
        super().__init__(text_only, conversation_format, format_prompt)
        self._single_image = "image" in dataset.features
        self._interleave_images = interleave_images

    def make_conversations(self, batch: dict) -> list[dict]:
        if self._single_image:
            return [
                make_image_conversation(
                    batch["image"][index],
                    batch["question"][index],
                    self._conversation_format,
                    self._format_prompt,
                )
                for index in range(len(batch["question"]))
            ]

        if not self._interleave_images:
            return [
                make_multi_image_conversation(
                    batch,
                    question,
                    index,
                    self._conversation_format,
                    self._format_prompt,
                )
                for index, question in enumerate(batch["question"])
            ]

        return [
            make_interleaved_image_conversation(
                batch,
                question,
                index,
                self._conversation_format,
                self._format_prompt,
            )
            for index, question in enumerate(batch["question"])
        ]

    def get_problem_ids(self, batch: dict) -> list[str]:
        return batch["id"]

    def get_questions(self, batch: dict) -> list[str]:
        return batch["question"]

    def get_answers(self, batch: dict) -> list[str]:
        return [a.replace("(", "").replace(")", "") for a in batch["correct"]]

    def get_images(self, batch: dict) -> list[list[Image]]:
        if self._single_image:
            return [[img] for img in batch["image"]]

        if not self._interleave_images:
            image_cols = sorted(
                [col for col in batch.keys() if col.startswith("image")]
            )
            return [
                [batch[col][index] for col in image_cols]
                for index in range(len(batch["question"]))
            ]

        return get_images_using_tags(batch, batch["question"])

    def get_image_columns(self) -> list[str]:
        return [f"image_{i}" for i in range(1, 5)]
