from datasets import Dataset

from src.dataset_adapters.base import DatasetAdapter, ConversationFormat
from src.dataset_adapters.common import (
    make_image_conversation,
    make_multi_image_conversation,
    make_interleaved_image_conversation,
    get_images_using_tags,
)


class RemiAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        text_only: bool,
        conversation_format: ConversationFormat,
        format_prompt: str = "",
        interleave_images: bool = True,
    ):
        super().__init__(text_only, conversation_format, format_prompt)
        self._question_ids = {
            row["question"]: index
            for index, row in enumerate(dataset.to_iterable_dataset())
        }
        self._single_image = "image" in dataset.features
        self._interleave_images = interleave_images

    def make_conversations(self, batch: dict):
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

    def get_problem_ids(self, batch: dict):
        return [self._question_ids[question] for question in batch["question"]]

    def get_questions(self, batch: dict):
        return batch["question"]

    def get_answers(self, batch: dict):
        return batch["label"]

    def get_images(self, batch: dict):
        if self._single_image:
            return [[img] for img in batch["image"]]

        if not self._interleave_images:
            image_cols = sorted(
                [col for col in batch.keys() if col.startswith("image")]
            )
            return [
                [batch[col][i] for col in image_cols]
                for i in range(len(batch["question"]))
            ]

        return get_images_using_tags(batch, batch["question"])

    def get_image_columns(self):
        return [f"image_{i}" for i in range(1, 7)]
