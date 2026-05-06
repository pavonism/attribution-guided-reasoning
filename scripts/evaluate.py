from datasets import Dataset, load_dataset, load_from_disk
import argparse
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation import GenerationMixin
from transformers.processing_utils import ProcessorMixin
from tqdm import tqdm

from math import ceil
from pathlib import Path

from src.formatting import AutoFormatter
from src.formatting.base import OutputFormatter
from src.model.evaluate import EvaluatedProblem, EvaluateResults
from src.dataset_adapters import ConversationFormat, DatasetAdapter, AutoDatasetAdapter
from src.v1 import V1ForConditionalGeneration, get_processor
from qwen_vl_utils import process_vision_info

transformers.utils.logging.set_verbosity_info()


def evaluate(
    processor: ProcessorMixin,
    model: GenerationMixin,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    output: str,
    batch_size: int,
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

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i in tqdm(batch_indexes):
        idxs = not_evaluated_job_indexes[i : i + batch_size]
        batch = dataset[idxs]

        problem_ids = dataset_adapter.get_problem_ids(batch)
        questions = dataset_adapter.get_questions(batch)
        answers = dataset_adapter.get_answers(batch)
        conversations = dataset_adapter.make_conversations(batch)
        images = [process_vision_info(c)[0] for c in conversations]

        text = processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        print(text)

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
            "repetition_penalty": 1.1,
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

        for j, response in enumerate(responses):
            print(response)

            extracted_answer = formatter.parse_answer(response)
            extracted_thought = formatter.parse_thoughts(response)

            solution = answers[j]
            is_correct = solution.lower() in extracted_answer.lower()
            if is_correct:
                correct += 1
            total += 1

            results.evaluations.append(
                EvaluatedProblem(
                    id=problem_ids[j],
                    problem=questions[j],
                    model_output=response,
                    extracted_answer=extracted_answer,
                    extracted_thought=extracted_thought,
                    solution=solution,
                    correct=is_correct,
                )
            )

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
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
        "--text_only",
        help="Evaluate using text-only mode without images",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--formatter",
        help="Formatter to use for parsing model output. 'vision-r1' or 'v1'",
        type=str,
        default="vision-r1",
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

    print("Loading model...")
    model = (
        V1ForConditionalGeneration.from_pretrained(
            args.model,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
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
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = 3136

    print("Loading dataset...")
    dataset = (
        load_from_disk(args.dataset)
        if Path(args.dataset).exists()
        else load_dataset(args.dataset, split=args.split)
    )

    formatter = AutoFormatter.from_name(args.formatter)

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        args.dataset,
        dataset,
        args.text_only,
        conversation_format=ConversationFormat.TRANSFORMERS,
        format_prompt=formatter.get_format_prompt(),
    )

    print("Running experiment...")
    evaluate(
        processor,
        model,
        dataset,
        dataset_adapter,
        args.output_file,
        args.batch_size,
        formatter,
        args.job_id,
        args.job_count,
        args.n_problems,
        args.parallel,
    )


if __name__ == "__main__":
    main()
