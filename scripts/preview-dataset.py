import os
from datasets import Dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description="Score the evaluation outputs.")
    parser.add_argument(
        "--path",
        help="Path to dataset.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--count",
        help="Show the number of problems in the dataset.",
        action="store_true",
    )
    parser.add_argument(
        "--head",
        help="Show only the first few examples of the dataset.",
        action="store_true",
    )
    parser.add_argument(
        "--problem_id",
        help="ID of the problem to show.",
        type=int,
    )

    args = parser.parse_args()

    path = args.path
    head = args.head
    problem_id = args.problem_id
    count = args.count

    if not os.path.exists(path):
        print(f"Provided path: {path} does not exist.")

    dataset = Dataset.load_from_disk(path)
    image_columns = [col for col in dataset.column_names if col.startswith("image")]
    dataset = dataset.remove_columns(image_columns)

    if count:
        print(f"Dataset has {len(dataset)} records in total.")

    if head:
        for i in range(5):
            record = dataset[i]
            print(record)

    if problem_id is not None:
        found_record = [record for record in dataset if record["id"] == problem_id][0]
        print(found_record)


if __name__ == "__main__":
    main()
