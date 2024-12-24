# Analysis of User Complaints and Problems Extracted from Social Media Posts Using NLP and Machine Learning
## Project Description:

The project demonstrates the process of automated analysis of texts from social networks, including data collection, processing, annotation, model training, and clustering. The tools and approaches used can be adapted and reused for working with texts from various sources. The project code offers flexibility for customization to specific tasks, such as data collection, text processing, interaction with the GPT API, and training custom models.

The following sections will provide a detailed overview of the project's structure, including its main components and a brief description of the associated files. Each component includes not only the scripts required for execution but also a Jupyter Notebook file. These notebooks provide a detailed guide with comments on how the created files and modules were utilized and executed, providing clarity and ease of replication.

## Project Structure

### 1. Data Collection Using VK API (api_post_collector)

#### Functionality:

- Collecting posts from specified VK groups where users share their problems.
- Adaptable for other platforms that support APIs (e.g., Telegram, Twitter).
- Easily extendable for various user content collection tasks.

#### Files
- vk_scraper.ipynb 
- vk_scraper.py #Uses the class from __vk_group.py__ to implement data collection functionality.
  - vk_group.py #Contains a class for interacting with the VK API and collecting data. 
- vk_json_to_csv.py #Performs a full cycle of processing JSON files collected using __vk_scraper.py__ and converts them into CSV format for further analysis. Utilizes methods from post_processing.py to preprocess textual data.
  - post_processing.py #Contains methods for basic text preprocessing, including removing empty posts, posts with only photos or links, emojis, advertisements, and filtering out overly long or short posts.

### 2. Text Labeling Using GPT API (gpt_text_processor)

#### Functionality:
- Extracts key problems from long texts and turns them into lists.
- Lets you modify API requests for different tasks.
- Suitable for classification, summarization, translation, and text correction.

#### Files

- data_labeling.ipynb
- data_labeling.py #Implements data labeling using GPT via __parallel_post_processor.py__
  - parallel_post_processor.py #Enables multiprocessing for data labeling to accelerate processing of large datasets.
     - open_ai.py #Provides a class for interacting with the GPT API to send requests.

### 3. Training a Model Based on ruT5-base (seq2seq_training)

After labeling the data, a custom model based on ruT5-base is created to perform the same labeling task.

#### Functionality:

- Trains ruT5-base on data labeled by GPT to handle the same task.
- The code for fine-tuning the model can be reused for other NLP tasks.
- Supports any Hugging Face-compatible models and can be adapted to different languages and tasks.

#### Files

- T5_Fine_Tuning.ipynb #Demonstrates the use of the fine-tuned model, including evaluation metrics for assessing the results of model distillation.
- train_seq2seq_model.py #Script for fine-tuning the T5 model on labeled data.

### 4. Clustering of Problems (Bonus Stage)

#### Functionality:

- Algorithms for clustering common problems have been implemented.
- Visualization of results has been implemented.
- Various clustering methods have been compared, and their features evaluated.
- The approach is suitable for clustering texts, topics, queries, and other data.

#### Files

- user_problem_clustering.ipynb #Explores clustering of processed data to identify and group common user problems, analyzing the structure using various clustering methods.
    - cluster_analysis.py #Contains tools for visualizing clustering results and additional utility functions to support the analysis.
  
