import re
from src.dataset_adapters.base import ConversationFormat


def make_image_content(img, format: ConversationFormat):
    if format == ConversationFormat.VLLM:
        return {
            "type": "image_pil",
            "image_pil": img,
        }
    elif format == ConversationFormat.TRANSFORMERS:
        return {"type": "image", "image": img}
    else:
        raise NotImplementedError(f"Format not supported: {format}")


def make_text_content(text: str):
    return {"type": "text", "text": text}


def make_image_conversation(
    img,
    question: str,
    format: ConversationFormat,
    format_prompt: str = "",
):
    if format_prompt:
        question += "\n\n" + format_prompt

    return [
        {
            "role": "user",
            "content": [
                make_image_content(img, format),
                make_text_content(question),
            ],
        },
    ]


def make_multi_image_conversation(
    batch: dict[str, list],
    question: str,
    batch_index: int,
    format: ConversationFormat,
    format_prompt: str = "",
):
    image_cols = sorted([col for col in batch.keys() if col.startswith("image")])
    images = [batch[col][batch_index] for col in image_cols]

    if format_prompt:
        question += "\n\n" + format_prompt

    return [
        {
            "role": "user",
            "content": [make_image_content(img, format) for img in images]
            + [make_text_content(question)],
        },
    ]


def make_interleaved_image_conversation(
    batch: dict[str, list],
    question: str,
    batch_index: int,
    format: ConversationFormat,
    format_prompt: str = "",
):
    matches = re.findall("(<image([1-9])+>)", question)
    contents = []

    if format_prompt:
        question += "\n\n" + format_prompt

    for img_tag, img_index in matches:
        text_fragments = question.split(img_tag)
        contents.extend(
            [
                make_text_content(text_fragments[0]),
                make_image_content(batch[f"image_{img_index}"][batch_index], format),
            ]
        )
        question = text_fragments[1]

    contents.append(make_text_content(question))

    return [
        {
            "role": "user",
            "content": contents,
        },
    ]


def get_images_using_tags(batch: dict[list], questions: list[str]):
    image_batch = []

    for index, question in enumerate(questions):
        matches = re.findall("<image([1-9])+>", question)

        images = [batch[f"image_{img_index}"][index] for img_index in matches]
        image_batch.append(images)

    return image_batch
