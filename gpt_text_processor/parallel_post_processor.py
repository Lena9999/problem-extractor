import pandas as pd
import os
import multiprocessing
import numpy as np
import json
from open_ai import OpenAIClient


class ParallelPostProcessor:
    """
    Processes a dataset of text posts in parallel by batching and sending requests to the OpenAI API.
    Successful results are stored in separate files for each process, while errors are logged in a shared file.

    Functions:
        - process_and_record_request: Sends a request for a given post, logs the result, and handles errors.
        - process_dataset_in_parallel: Splits the dataset into batches and manages parallel processing of posts. Successful results are stored in separate files for each process, while errors are logged in a shared file.
        - process_batch: Processes a batch of posts within a single process.

    Attributes:
        api_key (str): The API key for authenticating requests to the OpenAI API.
        organization (str): The organization ID associated with the OpenAI API key.
        results_dir (str): The directory where the results will be saved.
        failed_file (str): The path to the file for logging failed requests.
        success_dir (str): The directory where successful results are stored.
        lock (multiprocessing.Lock): A lock to ensure safe concurrent access to shared resources (like the error log).
        progress_counter (multiprocessing.Value): A shared counter to track progress across multiple processes.
    """

    def __init__(self, api_key, organization, results_dir="post_processing_results"):
        self.results_dir = results_dir
        self.api_key = api_key
        self.organization = organization
        self.failed_file = os.path.join(
            self.results_dir, "failed_requests.csv")
        self.success_dir = os.path.join(
            self.results_dir, "successful_requests")

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if not os.path.exists(self.success_dir):
            os.makedirs(self.success_dir)

        # Create a lock and a progress counter as class attributes
        self.lock = multiprocessing.Lock()
        self.progress_counter = multiprocessing.Value("i", 0)

    def process_and_record_request(
        self,
        api_client,
        post_id,
        post_text,
        prompt_template,
        result_output_file,
        model="gpt-4o-mini",
        temperature=0.5,
    ):
        """
        Sends a request to the OpenAI API for a post, records the result,
        and logs errors if the request fails.
        """

        try:
            prompt_temp = prompt_template.format(text=post_text)

            result = api_client.send_chat_request(
                prompt=prompt_temp, model=model, temperature=temperature
            )

            result_dict = json.loads(result)
            problems = result_dict["problems"]
            gender = result_dict["gender"]
            age = result_dict["age"]

            with open(result_output_file, "a") as f:
                f.write(f"{post_id}; {problems}; {gender}; {age}\n")

        except Exception as e:
            error_message = f"Error: {str(e)}, id: {post_id}, Model: {
                model}, Temperature: {temperature}\n"
            with self.lock:
                with open(self.failed_file, "a") as ff:
                    ff.write(error_message)

    def process_dataset_in_parallel(
        self,
        dataset,
        prompt_template,
        model="gpt-4o-mini",
        temperature=0.5,
        num_processes=4,
    ):
        """
        Args:
            dataset (pd.DataFrame): The dataset containing posts with 'id' and 'posts' columns.
            prompt_template (str): Template for formatting the API request.
                              Example:
                              "Analyze the following text: '{text}'"
            model (str, optional): The OpenAI model to use (default: 'gpt-4o-mini').
            temperature (float, optional): The temperature setting for the model (default: 0.5).
            num_processes (int, optional): The number of parallel processes to use (default: 4).
        """
        total_posts = len(dataset)

        post_batches = np.array_split(dataset, num_processes)

        processes = []
        for i, post_batch in enumerate(post_batches):
            result_output_file = os.path.join(
                self.success_dir, f"successful_requests_process_{i}.txt"
            )

            p = multiprocessing.Process(
                target=self.process_batch,
                args=(
                    post_batch,
                    prompt_template,
                    result_output_file,
                    total_posts,
                    model,
                    temperature,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    def process_batch(
        self,
        post_batch,
        prompt_template,
        result_output_file,
        total_posts,
        model="gpt-4o-mini",
        temperature=0.5,
        progress_interval=100,
    ):
        open_ai_client = OpenAIClient(
            api_key=self.api_key, organization=self.organization
        )
        for idx, row in post_batch.iterrows():
            post_text = row["posts"]
            post_id = row["id"]

            self.process_and_record_request(
                open_ai_client,
                post_id,
                post_text,
                prompt_template,
                result_output_file,
                model,
                temperature,
            )

            # Update progress every progress_interval posts
            with self.progress_counter.get_lock():
                self.progress_counter.value += 1
                if self.progress_counter.value % progress_interval == 0:
                    print(
                        f"Progress: {
                            self.progress_counter.value}/{total_posts} posts processed."
                    )

    def process_cluster(
        self,
        data,
        cluster_column,
        cluster_label,
        prompt_template,
        model,
        temperature,
        result_dict,
    ):
        """
        Args:
            data (pd.DataFrame): The dataset containing posts with "problems" column.
            cluster_column (str): Name of the column containing cluster labels.
            cluster_label (int/str): The label of the cluster to process.
            prompt_template (str): Template for formatting the API request.
            model (str): The OpenAI model to use.
            temperature (float): The temperature setting for the model.
            result_dict (dict): A shared dictionary to store results.
        """
        cluster_data = data[data[cluster_column] == cluster_label]

        cluster_texts = cluster_data["problems"].tolist()
        combined_text = ".".join(cluster_texts)

        open_ai_client = OpenAIClient(
            api_key=self.api_key, organization=self.organization
        )

        try:
            prompt = prompt_template.format(text=combined_text)

            result = open_ai_client.send_chat_request(
                prompt=prompt, model=model, temperature=temperature
            )

            result_dict[cluster_label] = result
        except Exception as e:
            result_dict["error"].append(cluster_label)

    def process_clusters_in_batches(
        self,
        data,
        cluster_column,
        prompt_template,
        batch_size,
        model="gpt-4o-mini",
        temperature=0.5,
    ):
        clusters = data[cluster_column].unique()
        cluster_batches = np.array_split(
            clusters, np.ceil(len(clusters) / batch_size))

        manager = multiprocessing.Manager()
        result_dict = manager.dict({"error": []})

        for batch in cluster_batches:
            processes = []
            for cluster_label in batch:
                p = multiprocessing.Process(
                    target=self.process_cluster,
                    args=(
                        data,
                        cluster_column,
                        cluster_label,
                        prompt_template,
                        model,
                        temperature,
                        result_dict,
                    ),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()
        return dict(result_dict)
