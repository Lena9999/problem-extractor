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

    def __init__(self, api_key, organization, results_dir='post_processing_results'):
        self.results_dir = results_dir
        self.api_key = api_key
        self.organization = organization
        self.failed_file = os.path.join(
            self.results_dir, 'failed_requests.csv')
        self.success_dir = os.path.join(
            self.results_dir, 'successful_requests')

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if not os.path.exists(self.success_dir):
            os.makedirs(self.success_dir)

        # Create a lock and a progress counter as class attributes
        self.lock = multiprocessing.Lock()
        self.progress_counter = multiprocessing.Value('i', 0)

    def process_and_record_request(self, api_client, post_id, post_text, prompt_template, result_output_file, model="gpt-4o-mini", temperature=0.5):
        """
        Sends a request to the OpenAI API for a post, records the result, 
        and logs errors if the request fails.

        Args:
            api_client (OpenAIClient): An instance of the OpenAI API client to send requests.
            post_id (str or int): The ID of the post to be processed.
            post_id (str or int): The ID of the post.
            post_text (str): The text of the post.
            prompt_template (str): Template for formatting the API request.
                              Example: 
                              "Analyze the following text: '{text}'"
            result_output_file (str): Path to the file for successful results.
        """

        try:
            # a prompt is being formed
            prompt_temp = prompt_template.format(text=post_text)

            result = api_client.send_chat_request(
                prompt=prompt_temp,
                model=model,
                temperature=temperature
            )

            result_dict = json.loads(result)
            problems = result_dict['problems']
            gender = result_dict['gender']
            age = result_dict['age']

            with open(result_output_file, 'a') as f:
                f.write(f"{post_id}; {problems}; {gender}; {age}\n")

        except Exception as e:
            # If an error occurs, write the request to the error file
            error_message = f"Error: {str(e)}, id: {post_id}, Model: {
                model}, Temperature: {temperature}\n"
            with self.lock:
                with open(self.failed_file, 'a') as ff:
                    ff.write(error_message)

    def process_dataset_in_parallel(self, dataset, prompt_template, model="gpt-4o-mini", temperature=0.5, num_processes=4):
        """
        Processes the dataset in parallel by splitting it into batches and distributing the work among multiple processes.
        Each process works on a batch of posts and stores the results in separate files for each process. Errors are logged in a shared file.

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

        # splitting data into batches
        post_batches = np.array_split(dataset, num_processes)

        processes = []
        for i, post_batch in enumerate(post_batches):
            # a file is generated for successful requests for each of the processes
            result_output_file = os.path.join(
                self.success_dir, f'successful_requests_process_{i}.txt')

            # processing each batch
            p = multiprocessing.Process(
                target=self.process_batch,
                args=(post_batch, prompt_template, result_output_file,
                      total_posts, model, temperature)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    def process_batch(self, post_batch, prompt_template, result_output_file, total_posts, model="gpt-4o-mini", temperature=0.5, progress_interval=100):
        """
        Processes a batch of posts in a single process.

        Args:
            post_batch (pd.DataFrame): A subset of the dataset containing posts with 'id' and 'posts' columns.
            prompt_template (str): Template for formatting the API request.
                              Example: "Analyze the following text: '{text}'".
            result_output_file (str): Path to the file where successful results for this batch will be recorded.
            total_posts (int): Total number of posts in the dataset (used for progress tracking).
            progress_interval (int, optional): The number of posts after which progress will be printed (default: 100).
        """
        open_ai_client = OpenAIClient(
            api_key=self.api_key, organization=self.organization)
        for idx, row in post_batch.iterrows():
            post_text = row['posts']
            post_id = row['id']

            self.process_and_record_request(
                open_ai_client, post_id, post_text, prompt_template, result_output_file, model, temperature)

            # Update progress every progress_interval posts
            with self.progress_counter.get_lock():
                self.progress_counter.value += 1
                if self.progress_counter.value % progress_interval == 0:
                    print(f"Progress: {
                          self.progress_counter.value}/{total_posts} posts processed.")
