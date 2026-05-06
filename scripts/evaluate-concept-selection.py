import random
from datasets import Dataset, load_dataset, load_from_disk
import argparse
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation import GenerationMixin
from transformers.processing_utils import ProcessorMixin
from tqdm import tqdm
from typing import Optional

from pathlib import Path

from src.dataset_adapters.base import ConversationFormat
from src.formatting import AutoFormatter
from src.formatting.base import OutputFormatter
from src.model.evaluate import (
    EvaluateResults,
    EvaluatedProblem,
)
from src.dataset_adapters import DatasetAdapter, AutoDatasetAdapter
from src.dataset_adapters.common import make_image_content, make_text_content
from src.masks import blend_mask
from src.v1 import V1ForConditionalGeneration, get_processor
from qwen_vl_utils import process_vision_info

transformers.utils.logging.set_verbosity_info()


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


def evaluate(
    processor: ProcessorMixin,
    model: GenerationMixin,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    output_dir: str,
    max_tokens: int,
    batch_size: int,
    formatter: OutputFormatter,
    background_visibility: float,
    n_concepts: int,
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
        not_evaluated_indexes = list(
            torch.randperm(len(not_evaluated_indexes))[
                : min(n_problems, len(not_evaluated_indexes))
            ]
        )

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

        # Create conversations
        conversations = [
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

        # Process images for Qwen format
        processed_images = [process_vision_info(c)[0] for c in conversations]

        text = processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=text,
            images=processed_images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "use_cache": True,
        }

        with torch.no_grad():
            input_ids_len = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, **gen_kwargs)
            generated_tokens = outputs[:, input_ids_len:]
            responses = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        for j, response_text in enumerate(responses):
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

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("Loading model...")
    model = (
        V1ForConditionalGeneration.from_pretrained(
            args.model,
            device_map="cuda",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        if "v1" in args.model
        else AutoModelForImageTextToText.from_pretrained(
            args.model,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    print("Loading processor...")
    processor = (
        get_processor(args.model)
        if "v1" in args.model
        else AutoProcessor.from_pretrained(args.model)
    )

    pad_token_id = processor.tokenizer.pad_token_id
    processor.pad_token_id = pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    if args.max_pixels:
        processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = 3136

    print("Loading dataset...")
    if Path(args.dataset).exists():
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    formatter = AutoFormatter.from_name(args.format)

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        args.dataset,
        dataset,
        text_only=False,
        conversation_format=ConversationFormat.TRANSFORMERS,
        format_prompt=formatter.get_format_prompt(),
    )

    background_visibility = (
        int(args.job_id) / (int(args.job_count) - 1)
        if args.job_id is not None and args.job_count is not None
        else args.background_visibility
    )

    print("Running evaluation...")
    evaluate(
        processor,
        model,
        dataset,
        dataset_adapter,
        args.output,
        args.max_tokens,
        args.batch_size,
        formatter,
        background_visibility,
        args.n_concepts,
        args.n_problems,
    )


if __name__ == "__main__":
    main()
