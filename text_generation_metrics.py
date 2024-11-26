from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import numpy as np
import math


class TextGenerationMetrics:
    def __init__(self):
        """
        A class for calculating various evaluation metrics to assess the quality of generated texts compared to reference texts. 

        This class provides methods to compute metrics such as BLEU, ROUGE, METEOR, precision, recall, F1, perplexity, and repetition rate.

        Methods:
            - calculate_bleu_scores(df, target_column, generated_column): Calculates BLEU scores for the dataset using sentence-level BLEU.
            - calculate_precision_recall_f1(df, target_column, generated_column): Computes precision, recall, and F1 scores based on token overlaps between reference and generated texts.
            - calculate_rouge_scores(df, target_column, generated_column): Calculates ROUGE-2 and ROUGE-L scores for the dataset.
            - calculate_meteor_scores(df, target_column, generated_column): Computes METEOR scores for the dataset.
            - calculate_perplexity(df, generated_column): Estimates the perplexity of the generated texts as a measure of text fluency.
            - calculate_repetition_rate(df, generated_column): Computes the repetition rate for generated texts, indicating how often tokens are repeated.
            - calculate_all_metrics(df, target_column, generated_column): Aggregates all metrics and computes their average values over the dataset.

        Parameters:
            df (pd.DataFrame): A pandas DataFrame containing reference texts and generated texts.
            target_column (str): The name of the column containing the reference (true) texts.
            generated_column (str): The name of the column containing the generated texts.

        """
        self.rouge = Rouge()

    def calculate_bleu_scores(self, df, target_column, generated_column):
        smoothing_function = SmoothingFunction().method1
        bleu_scores = []
        for target, generated in zip(df[target_column], df[generated_column]):
            reference = [target.split()]
            hypothesis = generated.split()
            score = sentence_bleu(reference, hypothesis,
                                  smoothing_function=smoothing_function)
            bleu_scores.append(score)
        return bleu_scores

    def calculate_precision_recall_f1(self, df, target_column, generated_column):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for target, generated in zip(df[target_column], df[generated_column]):
            target_labels = set(target.split())
            generated_labels = set(generated.split())

            true_positive = len(target_labels & generated_labels)
            false_positive = len(generated_labels - target_labels)
            false_negative = len(target_labels - generated_labels)

            precision = true_positive / \
                (true_positive + false_positive) if true_positive + \
                false_positive > 0 else 0
            recall = true_positive / \
                (true_positive + false_negative) if true_positive + \
                false_negative > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if precision + recall > 0 else 0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return precision_scores, recall_scores, f1_scores

    def calculate_rouge_scores(self, df, target_column, generated_column):
        rouge_2_scores = []
        rouge_l_scores = []
        for target, generated in zip(df[target_column], df[generated_column]):
            scores = self.rouge.get_scores(generated, target)[0]
            rouge_2_scores.append(scores["rouge-2"]["f"])
            rouge_l_scores.append(scores["rouge-l"]["f"])
        return rouge_2_scores, rouge_l_scores

    def calculate_meteor_scores(self, df, target_column, generated_column):
        meteor_scores = []
        for target, generated in zip(df[target_column], df[generated_column]):
            reference = target.split()
            hypothesis = generated.split()
            score = meteor_score([reference], hypothesis)
            meteor_scores.append(score)
        return meteor_scores

    def calculate_perplexity(self, df, generated_column):
        perplexities = []
        for generated in df[generated_column]:
            tokens = generated.split()
            log_probability = 0
            for token in tokens:
                log_probability += math.log(1 / len(tokens))
            perplexity = math.exp(-log_probability /
                                  len(tokens)) if tokens else float('inf')
            perplexities.append(perplexity)
        return perplexities

    def calculate_repetition_rate(self, df, generated_column):
        repetition_rates = []
        for generated in df[generated_column]:
            tokens = generated.split()
            if not tokens:
                repetition_rates.append(0)
                continue
            unique_tokens = len(set(tokens))
            repetition_rate = 1 - unique_tokens / len(tokens)
            repetition_rates.append(repetition_rate)
        return repetition_rates

    def calculate_all_metrics(self, df, target_column, generated_column):
        bleu_scores = self.calculate_bleu_scores(
            df, target_column, generated_column)
        precision_scores, recall_scores, f1_scores = self.calculate_precision_recall_f1(
            df, target_column, generated_column)
        rouge_2_scores, rouge_l_scores = self.calculate_rouge_scores(
            df, target_column, generated_column)
        meteor_scores = self.calculate_meteor_scores(
            df, target_column, generated_column)
        perplexities = self.calculate_perplexity(df, generated_column)
        repetition_rates = self.calculate_repetition_rate(df, generated_column)

        results = {
            "BLEU Score (average)": sum(bleu_scores) / len(bleu_scores),
            "Precision (average)": sum(precision_scores) / len(precision_scores),
            "Recall (average)": sum(recall_scores) / len(recall_scores),
            "F1 Score (average)": sum(f1_scores) / len(f1_scores),
            "ROUGE-2 (average)": sum(rouge_2_scores) / len(rouge_2_scores),
            "ROUGE-L (average)": sum(rouge_l_scores) / len(rouge_l_scores),
            "METEOR (average)": sum(meteor_scores) / len(meteor_scores),
            "Perplexity (average)": sum(perplexities) / len(perplexities),
            "Repetition Rate (average)": sum(repetition_rates) / len(repetition_rates),
        }

        return results


# Example usage:
# df = pd.DataFrame({'true_text': [...], 'predicted_text': [...]})
# metrics_calculator = TextGenerationMetrics()
# metrics = metrics_calculator.calculate_all_metrics(df, target_column='true_text', generated_column='predicted_text')
# print(metrics)
