import os
import argparse
import pandas as pd
from post_processing import PostProcessor


def main(args):
    try:
        print(f"Loading data from {args.input_file}...")
        df = pd.read_json(args.input_file)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    df.columns = args.columns
    preprocessor = PostProcessor(
        df,
        post_column=args.columns[1],
        min_length=args.min_length,
        max_length=args.max_length,
    )
    processed_data = preprocessor.preprocess(steps=args.steps)

    processed_csv_path = os.path.join(os.getcwd(), args.output_file)
    processed_data.to_csv(processed_csv_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess VK dataset scraped from a public group."
    )
    parser.add_argument("input_file", type=str,
                        help="Path to the input JSON file.")
    parser.add_argument("output_file", type=str,
                        help="Path to the output CSV file.")
    parser.add_argument(
        "--columns",
        type=str,
        nargs=3,
        default=["id", "posts", "Date"],
        help="Column names for the dataset: id, posts, and Date.",
    )
    parser.add_argument(
        "--min_length", type=int, default=15, help="Minimum length of a post."
    )
    parser.add_argument(
        "--max_length", type=int, default=300, help="Maximum length of a post."
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="*",
        default=None,
        help="List of preprocessing steps to execute. Possible values: ['remove_empty_posts', 'remove_posts_with_links', 'remove_duplicates', 'filter_by_length', 'remove_emojis']. If not provided, all steps are executed.",
    )

    args = parser.parse_args()
    main(args)
