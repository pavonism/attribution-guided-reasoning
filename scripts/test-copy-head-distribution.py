import argparse

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.generation import GenerationMixin
from datasets import Dataset, load_dataset, load_from_disk
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.v1 import V1ForConditionalGeneration, get_processor
from src.dataset_adapters import ConversationFormat, AutoDatasetAdapter, DatasetAdapter
from src.model.test_copy_head_distribution import ExperimentResult, ExperimentResults

transformers.utils.logging.set_verbosity_info()


def generate_response(
    processor: ProcessorMixin,
    model: GenerationMixin,
    conversation: list,
    images: list,
) -> str:
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": 8192,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "use_cache": True,
    }

    with torch.no_grad():
        input_ids_len = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs, **gen_kwargs)
        generated_tokens = outputs[:, input_ids_len:]
        response = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    return response


def evaluate_single_image(
    processor: ProcessorMixin,
    model: GenerationMixin,
    n_problems: int,
    results: ExperimentResults,
    idx: int,
    problem_id: str,
    image_cols: list[str],
    images: list,
    evaluated_problems: set[tuple[str, str]],
    output_path: Path | None = None,
    concatenated_image: bool = False,
):
    for image_col, image in zip(image_cols, images):
        if (problem_id, image_col) in evaluated_problems:
            continue

        user_message = (
            "What does the image depict?"
            if not concatenated_image
            else "What do the images depict?"
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_message},
                ],
            }
        ]
        response = generate_response(processor, model, conversation, [image])

        print("=" * 80)
        print(f"\n[Problem {idx + 1}/{n_problems} (ID: {problem_id}) - {image_col}]")
        print("=" * 80)
        print(f"Question: {user_message}")
        print(f"Response: {response}")

        results.results.append(
            ExperimentResult(
                problem_id=problem_id,
                image_cols=[image_col],
                prompt=user_message,
                response=response,
            )
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(results.model_dump_json())


def evaluate_multi_image(
    processor: ProcessorMixin,
    model: GenerationMixin,
    n_problems: int,
    results: ExperimentResults,
    idx: int,
    problem_id: str,
    image_cols: list[str],
    images: list,
    evaluated_problems: set[tuple[str, str]],
    output_path: Path | None = None,
):
    if (problem_id, image_cols[0]) in evaluated_problems:
        return

    user_message = "What do all images depict?"
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": user_message}],
        }
    ]
    response = generate_response(processor, model, conversation, images)

    print("=" * 80)
    print(f"\n[Problem {idx + 1}/{n_problems} (ID: {problem_id})]")
    print("=" * 80)
    print(f"Question: {user_message}")
    print(f"Response: {response}")

    results.results.append(
        ExperimentResult(
            problem_id=problem_id,
            image_cols=image_cols,
            prompt=user_message,
            response=response,
        )
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(results.model_dump_json())


def run_experiment(
    processor: ProcessorMixin,
    model: GenerationMixin,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    n_problems: int,
    output_path: Path | None = None,
    single_image: bool = False,
    concatenated_image: bool = False,
) -> ExperimentResults:
    results = ExperimentResults(results=[])
    n_problems = min(n_problems, len(dataset))

    if output_path and output_path.exists():
        print(f"Resuming from {output_path}")
        results = ExperimentResults.model_validate_json(output_path.read_text())

    evaluated_problems = {
        (result.problem_id, result.image_cols[0]) for result in results.results
    }

    weights = np.array(dataset["weight"])
    picked_problem_indexes = np.random.choice(
        list(range(len(dataset))),
        n_problems,
        p=weights / weights.sum(),
        replace=False,
    )

    for idx in tqdm(picked_problem_indexes):
        sample = dataset[idx : idx + 1]
        problem_id = dataset_adapter.get_problem_ids(sample)[0]

        image_cols = dataset_adapter.get_image_columns()
        images = dataset_adapter.get_images(sample)[0]

        if single_image:
            evaluate_single_image(
                processor,
                model,
                n_problems,
                results,
                idx,
                problem_id,
                image_cols,
                images,
                evaluated_problems,
                output_path,
                concatenated_image=concatenated_image,
            )
        else:
            evaluate_multi_image(
                processor,
                model,
                n_problems,
                results,
                idx,
                problem_id,
                image_cols,
                images,
                evaluated_problems,
                output_path,
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model's object detection in single and multi-image scenarios."
    )
    parser.add_argument("--model", help="Name or path to model", required=True)
    parser.add_argument("--dataset", help="Path to test dataset", required=True)
    parser.add_argument(
        "--n-problems",
        type=int,
        default=100,
        help="Number of problems to evaluate",
    )
    parser.add_argument(
        "--max-pixels",
        help="Maximum number of pixels for image processing",
        type=int,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--single-image",
        action="store_true",
        help="Evaluate single-image scenario (multi-image by default)",
    )
    parser.add_argument(
        "--concatenated-image",
        action="store_true",
        help="Evaluate multi-image scenario with single concatenated image instead of separate images.",
    )
    parser.add_argument(
        "--output",
        help="Path to output the evaluation results",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

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
            dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    print("Loading processor...")
    processor = (
        get_processor(args.model)
        if "v1" in args.model
        else AutoProcessor.from_pretrained(args.model)
    )
    if args.max_pixels:
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.min_pixels = 3136

    print("Loading dataset...")
    dataset = (
        load_from_disk(args.dataset)
        if Path(args.dataset).exists()
        else load_dataset(args.dataset, split="test")
    )

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        args.dataset,
        dataset,
        text_only=False,
        conversation_format=ConversationFormat.TRANSFORMERS,
    )

    print("Running evaluation...")
    output_path = Path(args.output) if args.output else None

    results = run_experiment(
        processor,
        model,
        dataset,
        dataset_adapter,
        args.n_problems,
        output_path,
        args.single_image,
        args.concatenated_image,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(results.model_dump_json())

    print(f"\nCompleted evaluation of {len(results.results)} results")


if __name__ == "__main__":
    main()
