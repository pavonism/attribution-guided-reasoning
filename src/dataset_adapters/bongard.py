from typing import List
from PIL.Image import Image
from datasets import Dataset
import numpy as np

from src.dataset_adapters.base import DatasetAdapter, ConversationFormat
from src.dataset_adapters.common import make_image_content, make_text_content


class BongardAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        text_only: bool,
        conversation_format: ConversationFormat,
        n_captions: int = 0,
        format_prompt: str = "",
    ):
        super().__init__(text_only, conversation_format, format_prompt)
        self._multi_image = "image" not in dataset.features
        self._text_only = text_only
        self.n_captions = n_captions

    def make_conversations(self, batch: dict):
        return [
            self._make_conversation(batch, index)
            for index in range(len(batch["problem"]))
        ]

    def get_problem_ids(self, batch: dict):
        return batch["id"]

    def get_questions(self, batch: dict):
        return batch["problem"]

    def get_answers(self, batch: dict):
        return batch["solution"]

    def get_images(self, batch: dict) -> list[list[Image]]:
        if self._multi_image:
            return self._get_images_for_multi_image(batch)
        else:
            return [[img] for img in batch["image"]]

    def get_image_columns(self):
        return (
            [f"image_left_{i}" for i in range(6)]
            + [f"image_right_{i}" for i in range(6)]
            if self._multi_image
            else ["image"]
        )

    def _get_images_for_multi_image(self, batch: dict) -> list[list[Image]]:
        image_batch = []

        for i in range(len(batch["problem"])):
            images = []
            for side in ["left", "right"]:
                for j in range(6):
                    images.append(batch[f"image_{side}_{j}"][i])

            image_batch.append(images)

        return image_batch

    def _make_conversation(
        self,
        batch: dict,
        index: int,
    ):
        problem = batch["problem"][index]

        if self._format_prompt:
            problem += "\n\n" + self._format_prompt

        if self._text_only:
            return self._make_conversation_text(problem)
        if self._multi_image:
            left_images = self._get_side_images("left", batch, index)
            right_images = self._get_side_images("right", batch, index)
            left_captions = self._get_side_captions("left", batch, index)
            right_captions = self._get_side_captions("right", batch, index)

            return self._make_conversation_multi_image(
                problem,
                left_images,
                right_images,
                left_captions,
                right_captions,
            )

        img = batch["image"][index]
        return self._make_conversation_image(problem, img)

    def _images_to_contents(self, images: List[Image], captions: List[str]):
        contents = []
        picked_captions = np.random.choice(
            len(captions),
            min(self.n_captions, len(captions)),
            replace=False,
        )

        for i, img in enumerate(images):
            contents.extend(
                [
                    make_text_content(f"Image #{i + 1}"),
                    make_image_content(img, self._conversation_format),
                ]
            )

            if i in picked_captions:
                caption = captions[i]
                contents.append(make_text_content(caption))

        return contents

    def _make_conversation_multi_image(
        self,
        problem: str,
        left_images: List[Image],
        right_images: List[Image],
        left_captions: List[str],
        right_captions: List[str],
    ):
        left_images_contents = self._images_to_contents(left_images, left_captions)
        right_images_contents = self._images_to_contents(right_images, right_captions)

        return [
            {
                "role": "user",
                "content": [
                    make_text_content("\n\nLeft:"),
                    *left_images_contents,
                    make_text_content("\n\nRight:"),
                    *right_images_contents,
                    make_text_content(problem),
                ],
            },
        ]

    def _make_conversation_image(self, problem: str, img: Image):
        return [
            {
                "role": "user",
                "content": [
                    make_image_content(img, self._conversation_format),
                    make_text_content(problem),
                ],
            },
        ]

    def _make_conversation_text(self, problem: str):
        return [
            {
                "role": "user",
                "content": [make_text_content(problem)],
            },
        ]

    def _get_side_images(self, side: str, batch: dict, index: int):
        images = []

        for j in range(6):
            column_name = f"image_{side}_{j}"

            if column_name in batch:
                img = batch[column_name][index]
                images.append(img)

        return images

    def _get_side_captions(self, side: str, batch: dict, index: int):
        captions = []

        for j in range(6):
            column_name = f"caption_{side}_{j}"

            if column_name in batch:
                caption = batch[column_name][index]
                captions.append(caption)

        return captions
