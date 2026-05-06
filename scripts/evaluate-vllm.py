import os
from datasets import Dataset, load_dataset, load_from_disk
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from math import ceil
from src.masks import adjust_to_patch_size

from pathlib import Path

from src.dataset_adapters.base import ConversationFormat
from src.formatting import AutoFormatter
from src.formatting.base import OutputFormatter
from src.model.evaluate import EvaluatedProblem, EvaluateResults
from src.dataset_adapters import DatasetAdapter, AutoDatasetAdapter


def evaluate(
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    output: str,
    batch_size: int,
    max_tokens: int,
    max_pixels: int,
    patch_size: int,
    merge_size: int,
    formatter: OutputFormatter,
    job_id: int,
    job_count: int,
    n_problems: int,
    parallel: bool,
):
    output_file = Path(output) / f"{job_id}_of_{job_count}.json" if parallel else output

    correct = 0
    total = 0
    results = EvaluateResults(evaluations=[])

    if Path(output_file).exists():
        results = EvaluateResults.from_file(output_file)

    evaluated_ids = set(evaluation.id for evaluation in results.evaluations)

    all_problem_ids = dataset_adapter.get_problem_ids(dataset)
    problem_ids_to_evaluate = all_problem_ids[:n_problems]

    job_problem_ids_count = ceil(len(problem_ids_to_evaluate) / job_count)
    job_indexes = list(
        range(
            (job_id - 1) * job_problem_ids_count,
            min(job_id * job_problem_ids_count, len(problem_ids_to_evaluate)),
        )
    )

    not_evaluated_job_indexes = [
        i for i in job_indexes if problem_ids_to_evaluate[i] not in evaluated_ids
    ]

    batch_indexes = list(range(0, len(not_evaluated_job_indexes), batch_size))
    sampling_params = SamplingParams(max_tokens=max_tokens)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, batch_idx in enumerate(tqdm(batch_indexes)):
        batch_idxs = not_evaluated_job_indexes[batch_idx : batch_idx + batch_size]

        batch = dataset[batch_idxs]

        problem_ids = dataset_adapter.get_problem_ids(batch)
        questions = dataset_adapter.get_questions(batch)
        answers = dataset_adapter.get_answers(batch)
        conversations = dataset_adapter.make_conversations(batch)
        images = dataset_adapter.get_images(batch)

        images = [
            [
                adjust_to_patch_size(image, patch_size, merge_size, max_pixels)
                for image in image_list
            ]
            for image_list in images
        ]

        inputs = [
            {
                "prompt": tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "multi_modal_data": {"image": image_list},
            }
            for conversation, image_list in zip(conversations, images)
        ]

        responses = model.generate(inputs, sampling_params)

        for j, response in enumerate(responses):
            output = response.outputs[0].text

            extracted_answer = formatter.parse_answer(output)
            extracted_thought = formatter.parse_thoughts(output)

            solution = answers[j]
            is_correct = solution.lower() in extracted_answer.lower()
            if is_correct:
                correct += 1
            total += 1

            results.evaluations.append(
                EvaluatedProblem(
                    id=problem_ids[j],
                    problem=questions[j],
                    model_output=output,
                    extracted_answer=extracted_answer,
                    extracted_thought=extracted_thought,
                    solution=solution,
                    correct=is_correct,
                )
            )

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

        if i % 2 == 0:
            results.to_file(output_path)

    results.to_file(output_path)
    print(f"Evaluation results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Bongard-RWR+ data.")
    parser.add_argument("--model", help="Name or path to model")
    parser.add_argument("--dataset", help="Path to test dataset")
    parser.add_argument("--output_file", help="Path to evaluation output file")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_pixels",
        help="Maximum number of pixels for image processing",
        type=int,
    )
    parser.add_argument(
        "--tensor_parallel_size",
        help="Number of GPUs to use for tensor parallelism",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--text_only",
        help="Evaluate using text-only mode without images",
        action="store_true",
    )
    parser.add_argument(
        "--max-tokens",
        help="Maximum number of tokens to generate",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--formatter",
        help="Formatter to use for model output. Options: 'default', 'html', 'markdown'",
        default="vision-r1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split",
        help="Dataset split to evaluate on, if the dataset has splits.",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        help="ID of the current job for distributed evaluation (if applicable).",
        default=1,
    )
    parser.add_argument(
        "--job-count",
        type=int,
        help="Total number of jobs for distributed evaluation (if applicable).",
        default=1,
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=-1,
        help="Maximum number of problems to evaluate (useful for testing).",
    )
    parser.add_argument(
        "--parallel",
        help="Whether to use parallel processing for evaluation.",
        action="store_true",
    )

    args = parser.parse_args()
    print(args)

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

    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    formatter = AutoFormatter.from_name(args.formatter)

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        args.dataset,
        dataset,
        args.text_only,
        format_prompt=formatter.get_format_prompt(),
        conversation_format=ConversationFormat.TRANSFORMERS,
    )

    patch_size = 16 if "qwen3" in args.model.lower() else 14
    merge_size = 2

    evaluate(
        model,
        tokenizer,
        dataset,
        dataset_adapter,
        args.output_file,
        args.batch_size,
        args.max_tokens,
        args.max_pixels,
        patch_size,
        merge_size,
        formatter,
        args.job_id,
        args.job_count,
        args.n_problems,
        args.parallel,
    )


if __name__ == "__main__":
    main()
