import os
import json
import pandas as pd
from parallel_post_processor import ParallelPostProcessor
import argparse
from datetime import datetime


def load_prompt(prompt_path):
    with open(prompt_path, "r") as file:
        return file.read()


def save_labeled_data(folder_path, column_names, num_processes):
    data_set_labeled = pd.DataFrame(columns=column_names)

    successful_requests_folder = os.path.join(
        folder_path, "successful_requests")

    for i in range(num_processes):
        file_path = os.path.join(
            successful_requests_folder, f"successful_requests_process_{i}.txt"
        )
        if os.path.exists(file_path):
            current_process = pd.read_csv(
                file_path, delimiter=";", header=None)
            current_process.columns = column_names
            data_set_labeled = pd.concat(
                [data_set_labeled, current_process], ignore_index=True
            )
        else:
            print(f"Warning: File {file_path} not found.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        folder_path, f"final_labeled_dataset_{timestamp}.csv")

    data_set_labeled.to_csv(output_file, index=False, sep=";")
    print(f"Final dataset saved to: {output_file}")


def main(
    config_path,
    dataset_path,
    labeled_data_folders,
    prompt_path,
    model,
    temperature,
    num_processes,
):
    dataset = pd.read_csv(dataset_path)

    with open(config_path, "r") as json_file:
        open_ai_data = json.load(json_file)

    parallel_p_p = ParallelPostProcessor(
        results_dir=labeled_data_folders[0],
        api_key=open_ai_data["api_key"],
        organization=open_ai_data["organization"],
    )

    prompt = load_prompt(prompt_path)
    parallel_p_p.process_dataset_in_parallel(
        dataset=dataset,
        prompt_template=prompt,
        model=model,
        temperature=temperature,
        num_processes=num_processes,
    )

    column_names = ["id", "problems", "gender", "age"]
    save_labeled_data(
        folder_path=labeled_data_folders[0],
        column_names=column_names,
        num_processes=num_processes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data labeling script for processing and combining labeled datasets."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to OpenAI configuration JSON file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the input dataset (CSV file).",
    )
    parser.add_argument(
        "--labeled_data_folders",
        type=str,
        nargs="+",
        required=True,
        help="Paths to folders with labeled data.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to the prompt template file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="The model to use for processing (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature parameter for model creativity (default: 0.5).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4).",
    )

    args = parser.parse_args()

    main(
        config_path=args.config_path,
        dataset_path=args.dataset_path,
        labeled_data_folders=args.labeled_data_folders,
        prompt_path=args.prompt_path,
        model=args.model,
        temperature=args.temperature,
        num_processes=args.num_processes,
    )
