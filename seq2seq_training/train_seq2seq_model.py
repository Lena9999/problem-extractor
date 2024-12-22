import os
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm

# Отключаем WANDB
os.environ["WANDB_DISABLED"] = "true"


def tokenize_function(examples, tokenizer, max_input_length, max_target_length):
    inputs = tokenizer(
        examples["input_text"], max_length=max_input_length, truncation=True, padding="max_length"
    )
    targets = tokenizer(
        examples["target_text"], max_length=max_target_length, truncation=True, padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


def prepare_tokenized_dataset(df, tokenizer, max_input_length, max_target_length):
    dataset = Dataset.from_pandas(df)
    return dataset.map(lambda examples: tokenize_function(examples, tokenizer, max_input_length, max_target_length), batched=True)


def main(args):
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    dataset = pd.read_csv(args.dataset_path)

    if args.input_column not in dataset.columns:
        raise ValueError(f"Input column '{
                         args.input_column}' not found in dataset.")
    if args.target_column not in dataset.columns:
        raise ValueError(f"Target column '{
                         args.target_column}' not found in dataset.")

    dataset = dataset[[args.input_column, args.target_column, "id"]].dropna().rename(
        columns={args.input_column: 'input_text',
                 args.target_column: 'target_text'}
    )

    train_df, temp_df = train_test_split(
        dataset, test_size=args.test_size, random_state=42
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42
    )

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(valid_df)}")
    print(f"Test set size: {len(test_df)}")

    train_tokenized = prepare_tokenized_dataset(
        train_df, tokenizer, args.max_input_length, args.max_target_length)
    valid_tokenized = prepare_tokenized_dataset(
        valid_df, tokenizer, args.max_input_length, args.max_target_length)
    test_tokenized = prepare_tokenized_dataset(
        test_df, tokenizer, args.max_input_length, args.max_target_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        logging_steps=10,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(
            description="Train a T5 model for text generation.")
        parser.add_argument("--dataset_path", type=str,
                            required=True, help="Path to the dataset CSV file.")
        parser.add_argument("--output_dir", type=str, default="results",
                            help="Directory to save the model and results.")
        parser.add_argument("--model_name", type=str,
                            default="ai-forever/ruT5-base", help="Model name or path.")
        parser.add_argument("--num_epochs", type=int,
                            default=6, help="Number of training epochs.")
        parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size for training and evaluation.")
        parser.add_argument("--max_input_length", type=int,
                            default=200, help="Maximum length for input text.")
        parser.add_argument("--max_target_length", type=int,
                            default=60, help="Maximum length for target text.")
        parser.add_argument("--test_size", type=float, default=0.3,
                            help="Test set size as a fraction of the dataset.")
        parser.add_argument("--input_column", type=str, required=True,
                            help="Name of the input column in the dataset.")
        parser.add_argument("--target_column", type=str, required=True,
                            help="Name of the target column in the dataset.")
        return parser.parse_args()

    args = parse_arguments()
    main(args)
