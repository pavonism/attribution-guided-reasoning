import argparse
import os
from pathlib import Path
import numpy as np
from collections import Counter
import tqdm

from datasets import load_from_disk, load_dataset, Dataset

from src.dataset_adapters.base import DatasetAdapter
from src.model.evaluate import EvaluatedProblem, EvaluateResults
from src.virft.rewards import AutoReward, BaseReward

from src.dataset_adapters import AutoDatasetAdapter


def get_source_problem_id(evaluated_problem: EvaluatedProblem) -> int:
    return evaluated_problem.id % 1_000


def reassign_correct_answers(
    results: EvaluateResults,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
) -> EvaluateResults:
    reevaluated_evaluations = []

    problem_ids = dataset_adapter.get_problem_ids(dataset)
    solutions = dataset_adapter.get_answers(dataset)
    problem_id_to_solution = {
        pid: solution for pid, solution in zip(problem_ids, solutions)
    }

    for evaluation in tqdm.tqdm(results.evaluations):
        solution = problem_id_to_solution.get(evaluation.id, "Solution not found")

        print(f"Reevaluating problem {evaluation.id}...")
        print(f"Original solution: {evaluation.solution}")
        print(f"Reevaluated solution: {solution}")

        reevaluated_evaluation = EvaluatedProblem(
            id=evaluation.id,
            problem=evaluation.problem,
            model_output=evaluation.model_output,
            solution=solution,
            extracted_thought=evaluation.extracted_thought,
            extracted_answer=evaluation.extracted_answer,
            correct=False,
        )

        reevaluated_evaluations.append(reevaluated_evaluation)

    return EvaluateResults(evaluations=reevaluated_evaluations)


def get_mean_reward(
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    results: EvaluateResults,
    reward_func: BaseReward,
) -> float:
    rewards = []
    problem_id_to_evaluation = {
        evaluation.id: evaluation for evaluation in results.evaluations
    }

    for index in tqdm.tqdm(range(len(dataset))):
        batch = dataset[index : index + 1]

        problem_id = dataset_adapter.get_problem_ids(batch)[0]
        evaluation = problem_id_to_evaluation.get(problem_id)
        
        if evaluation is None:
            print(f"WARN: NO EVALUATION FOUND FOR PROBLEM_ID: {problem_id}")
            continue

        answers = dataset_adapter.get_answers(batch)

        batch["solution"] = [evaluation.solution]

        if evaluation:
            reward = reward_func(
                [
                    [
                        {
                            "content": evaluation.model_output,
                        }
                    ]
                ],
                patch_size=14 * 2,
                min_pixels=(14 * 2) ** 2,
                max_pixels=1568000,
                **batch,
            )
            rewards.append(reward[0])

    return sum(rewards) / len(results.evaluations)


def get_mean_outputs_length(results: EvaluateResults) -> float:
    outputs_lengths = [len(x.model_output) for x in results.evaluations]
    return np.mean(outputs_lengths)


def get_mean_thoughts_length(results: EvaluateResults) -> float:
    thoughts_lenghts = [len(x.extracted_thought) for x in results.evaluations]
    return np.mean(thoughts_lenghts)


def get_std_outputs_length(results: EvaluateResults) -> float:
    outputs_lengths = [len(x.model_output) for x in results.evaluations]
    return np.std(outputs_lengths)


def get_std_thoughts_length(results: EvaluateResults) -> float:
    thoughts_lenghts = [len(x.extracted_thought) for x in results.evaluations]
    return np.std(thoughts_lenghts)


def print_metrics_per_problem(results: EvaluateResults, reward_func):
    source_problem_ids = set(get_source_problem_id(x) for x in results.evaluations)
    metrics = []

    for problem_id in sorted(source_problem_ids):
        current_results = EvaluateResults(
            evaluations=[
                x for x in results.evaluations if get_source_problem_id(x) == problem_id
            ]
        )

        metrics.append(
            {
                "problem_id": problem_id,
                "solution": current_results.evaluations[0].solution,
                "accuracy": get_mean_reward(current_results, reward_func),
                "mean_outputs_length": get_mean_outputs_length(current_results),
                "mean_thoughts_length": get_mean_thoughts_length(current_results),
            },
        )

    print(metrics)


def get_top_n_answers(results: EvaluateResults, top_n: int):
    model_answers = [x.extracted_answer for x in results.evaluations]
    counter = Counter(model_answers)
    return counter.most_common(top_n)


def main():
    parser = argparse.ArgumentParser(description="Score the evaluation outputs.")
    parser.add_argument("--outputs_file", help="Path to outputs JSON file")
    parser.add_argument(
        "--dataset",
        help="Path to the dataset directory, when provided the answers are reevaluated against the ground truth.",
    )
    parser.add_argument(
        "--split",
        help="Dataset split",
    )
    parser.add_argument(
        "--format",
        help="Output format",
        type=str,
        default="vision-r1",
        choices=["vision-r1", "v1"],
    )
    parser.add_argument(
        "--per_problem",
        help="Print detailed metrics for each problem",
        action="store_true",
    )
    parser.add_argument(
        "--top_answers",
        help="Number of top most frequent model answers to display",
        type=int,
    )
    parser.add_argument(
        "--reward_func",
        help="Reward function to use for evaluation",
        type=str,
        default="string_matching",
        choices=["bow", "string_matching", "attribution"],
    )
    parser.add_argument(
        "--all-words",
        help="Include all words while computing the mean reward",
        action="store_true",
    )
    parser.add_argument(
        "--reassign-answers",
        help="Reassign correct answers based on the dataset and split",
        action="store_true",
    )

    args = parser.parse_args()
    outputs_file = args.outputs_file
    format = args.format
    per_problem = args.per_problem
    top_answers = args.top_answers
    reward_func = args.reward_func
    all_words = args.all_words

    print(f"Outputs file: {outputs_file}")
    print(f"Output format: {format}")
    print(f"Print metrics for each problem: {per_problem}")
    print(f"Top N answers: {top_answers}")
    print(f"Reward function: {reward_func}")
    print(f"All words: {all_words}")

    if os.path.isfile(outputs_file):
        results = EvaluateResults.from_file(outputs_file)
    else:
        files = list(Path(outputs_file).glob("*.json"))
        results = EvaluateResults(evaluations=[])
        for file in files:
            print(f"Loading results from {file}...")
            file_results = EvaluateResults.from_file(file)
            results.evaluations.extend(file_results.evaluations)

    dataset = (
        load_from_disk(args.dataset)
        if Path(args.dataset).exists()
        else load_dataset(args.dataset, split=args.split)
    )

    dataset_adapter: DatasetAdapter = AutoDatasetAdapter.from_dataset(
        args.dataset,
        dataset,
        False,
    )

    image_cols = [col for col in dataset.column_names if col.startswith("image")]

    if not "attribution" in reward_func: 
        dataset = dataset.remove_columns(image_cols)

    reward_func = AutoReward.from_name(reward_func, all_words=all_words, format=format)

    if args.dataset and args.reassign_answers:
        results = reassign_correct_answers(results, dataset, dataset_adapter)
        results.to_file(outputs_file)

    if per_problem:
        print_metrics_per_problem(results, reward_func)
        print("\n")

    print(
        f"Mean Reward: {get_mean_reward(dataset, dataset_adapter, results, reward_func):2f}"
    )
    print(
        f"Total Mean Outputs Length: {get_mean_outputs_length(results):2f}, std: {get_std_outputs_length(results):2f}"
    )
    print(
        f"Total Mean Thoughts Length: {get_mean_thoughts_length(results):2f}, std: {get_std_thoughts_length(results):2f}"
    )

    if top_answers:
        top_n_answers = get_top_n_answers(results, top_answers)
        print("\n")
        print("Top 5 most occuring model's outputs:")
        for answer, count in top_n_answers:
            print(count, answer)


if __name__ == "__main__":
    main()
