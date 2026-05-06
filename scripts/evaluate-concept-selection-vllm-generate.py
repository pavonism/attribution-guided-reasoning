import os
import random
from PIL.Image import Image
from datasets import Dataset, load_dataset, load_from_disk
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Optional
import numpy as np

from pathlib import Path

from src.configuration import get_vllm_sampling_params
from src.dataset_adapters.base import ConversationFormat
from src.dataset_adapters.common import make_text_content, make_image_content
from src.formatting import AutoFormatter
from src.formatting.base import OutputFormatter
from src.model.evaluate import (
    EvaluateResults,
    EvaluatedProblem,
)
from src.dataset_adapters import DatasetAdapter, AutoDatasetAdapter
from src.masks import blend_mask


def make_concept_selection_prompt(options: list[str]) -> str:
    options_text = "\n".join(options)

    prompt = f"""
The goal in solving a Bongard Problem is to identify a concept that differentiates the left and right sides. 
All images belonging to the LEFT side represent a common, shared concept which is not present in any image from the RIGHT side, 
and vice versa - all images belonging to the RIGHT side represent a common, shared concept which is not present in any image from the LEFT side.

From the following options, select the concept that is represented by the LEFT images.
{options_text}

Respond with only the letter (A, B, C, etc.) of the correct answer.""".strip()
    return prompt


def make_conversations(prompts: list[str], images: list[Image]):
    return [
        [
            {
                "role": "user",
                "content": [
                    make_image_content(image, ConversationFormat.TRANSFORMERS),
                    make_text_content(prompt),
                ],
            }
        ]
        for prompt, image in zip(prompts, images)
    ]


def evaluate(
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    output_dir: str,
    batch_size: int,
    sampling_params: SamplingParams,
    n_concepts: int,
    formatter: OutputFormatter,
    background_visibility: float,
    n_problems: Optional[int],
):
    """Evaluate model on concept selection task."""

    output_path = Path(
        output_dir, f"background-visibility-{background_visibility}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results = EvaluateResults(evaluations=[])

    if output_path.exists():
        print(f"Resuming evaluation from {output_path}")
        evaluation_results = EvaluateResults.from_file(output_path)

    all_problem_ids = dataset_adapter.get_problem_ids(dataset)
    all_answers = dataset_adapter.get_answers(dataset)

    evaluated_problem_ids = set(e.id for e in evaluation_results.evaluations)
    not_evaluated_indexes = [
        i
        for i in range(len(dataset))
        if all_problem_ids[i] not in evaluated_problem_ids
    ]

    if n_problems is not None:
        not_evaluated_indexes = np.random.choice(
            not_evaluated_indexes,
            min(n_problems, len(not_evaluated_indexes)),
            replace=False,
        ).tolist()

    total = 0
    correct = 0

    batch_indexes = list(range(0, len(not_evaluated_indexes), batch_size))

    for batch_idx in tqdm(batch_indexes):
        batch_idxs = not_evaluated_indexes[batch_idx : batch_idx + batch_size]

        batch = dataset[batch_idxs]

        problem_ids = dataset_adapter.get_problem_ids(batch)
        correct_answers = dataset_adapter.get_answers(batch)
        images = dataset_adapter.get_images(batch)
        masks = batch["mask"]

        images = [
            blend_mask(
                image[0].convert("RGBA"),
                mask.convert("RGBA"),
                background_visibility=background_visibility,
            )
            for image, mask in zip(images, masks)
        ]

        prompts, batch_correct_options = generate_concept_selection_prompts(
            n_concepts,
            all_answers,
            correct_answers,
            formatter.get_format_prompt(),
        )

        conversations = make_conversations(prompts, images)

        inputs = [
            {
                "prompt": tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "multi_modal_data": {"image": image},
            }
            for conversation, image in zip(conversations, images)
        ]

        responses = model.generate(
            inputs,
            sampling_params=sampling_params,
        )

        for j, response in enumerate(responses):
            response_text = response.outputs[0].text
            extracted_answer = formatter.parse_answer(response_text).strip()
            correct_answer = batch_correct_options[j]
            is_correct = extracted_answer.upper() == correct_answer.upper()

            if is_correct:
                correct += 1
            total += 1

            evaluation_results.evaluations.append(
                EvaluatedProblem(
                    id=problem_ids[j],
                    problem=prompts[j],
                    solution=correct_answer,
                    model_output=response_text,
                    extracted_answer=extracted_answer,
                    extracted_thought="",
                    correct=is_correct,
                )
            )

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        evaluation_results.to_file(output_path)

    print(f"Evaluation results saved to {output_path}")


def generate_concept_selection_prompts(
    n_concepts: int,
    all_answers: list[str],
    correct_answers: list[str],
    format_prompt: str,
) -> tuple[list[str], list[str]]:
    prompts = []
    batch_correct_answers = []

    for correct_answer in correct_answers:
        other_indexes = [
            i for i in range(len(all_answers)) if all_answers[i] != correct_answer
        ]
        random_distractors = random.sample(
            other_indexes, min(n_concepts - 1, len(other_indexes))
        )
        distractors = [all_answers[i] for i in random_distractors]

        options = [correct_answer] + distractors
        random.shuffle(options)

        letters = []

        for idx, option in enumerate(options):
            letter = chr(ord("A") + idx)
            letters.append(letter)

            if option == correct_answer:
                batch_correct_answers.append(letter)

        options = [f"{letter}. {option}" for letter, option in zip(letters, options)]

        prompt = make_concept_selection_prompt(options)

        prompt = f"{prompt}\n\n{format_prompt}" if format_prompt else prompt

        prompts.append(prompt)

    return prompts, batch_correct_answers


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on concept selection task."
    )
    parser.add_argument("--model", help="Name or path to model.")
    parser.add_argument("--dataset", help="Path to test dataset.")
    parser.add_argument("--output", help="Path to evaluation output file.")
    parser.add_argument(
        "--n-concepts",
        type=int,
        default=4,
        help="Number of concept options to choose from.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        help="Number of GPUs to use for tensor parallelism.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-tokens",
        help="Maximum number of tokens to generate.",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility.",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--format",
        help="Output format for parsing model responses.",
        type=str,
        default="vision-r1",
        choices=["vision-r1", "v1"],
    )
    parser.add_argument(
        "--max-pixels",
        help="Maximum number of pixels for image processing (if applicable).",
        type=int,
    )
    parser.add_argument(
        "--split",
        help="Dataset split to evaluate on, if the dataset has splits.",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--background-visibility",
        help="The visibility of the background when visualizing masks.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--job-id",
        help="ID of the current job for distributed evaluation (if applicable).",
        default=None,
    )
    parser.add_argument(
        "--job-count",
        help="Total number of jobs for distributed evaluation (if applicable).",
        default=None,
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate (useful for testing).",
    )

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)

    sampling_params = get_vllm_sampling_params(args.model)
    sampling_params.max_tokens = args.max_tokens

    model = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=16_384 + args.max_tokens,
        mm_processor_kwargs={
            "min_pixels": 3136,
            "max_pixels": args.max_pixels,
        },
        seed=args.seed,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    formatter = AutoFormatter.from_name(args.format)

    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        args.dataset,
        dataset,
        text_only=False,
        format_prompt=formatter.get_format_prompt(),
    )

    background_visibility = (
        int(args.job_id) / (int(args.job_count) - 1)
        if args.job_id is not None and args.job_count is not None
        else args.background_visibility
    )

    evaluate(
        model,
        tokenizer,
        dataset,
        dataset_adapter,
        args.output,
        args.batch_size,
        sampling_params,
        args.n_concepts,
        formatter,
        background_visibility,
        args.n_problems,
    )


if __name__ == "__main__":
    main()
