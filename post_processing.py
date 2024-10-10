import pandas as pd
from urlextract import URLExtract


class PostProcessor:
    def __init__(self, data, post_column='posts', min_length=15, max_length=3000):
        """
        A class used to preprocess datasets containing textual posts.

        This class offers several key preprocessing functionalities:
        1. Removing rows with empty posts.
        2. Removing rows containing links within the posts.
        3. Removing duplicate posts.
        4. Filtering posts by their length (minimum and maximum length).

        Parameters:
        data (pd.DataFrame): The dataset to preprocess.
        post_column (str): The name of the column that contains post texts. Defaults to 'posts'.
        """
        self.data = data
        self.extractor = URLExtract()  # Initialize object for URL extraction

        if post_column not in data.columns:
            raise ValueError(
                f"The column '{post_column}' does not exist in the dataset.")

        self.post_column = post_column
        self.min_length = min_length
        self.max_length = max_length
        self.cleaned_data = None

    def remove_empty_posts(self):
        """Removes rows with empty posts."""
        original_size = len(self.data)
        self.cleaned_data = self.data.dropna(subset=[self.post_column]).copy()
        cleaned_size = len(self.cleaned_data)
        removed_posts = original_size - cleaned_size
        print(f"Number of empty posts removed: {removed_posts}")
        return self.cleaned_data

    def remove_posts_with_links(self):
        """Removes posts containing at least one link."""
        original_size = len(self.cleaned_data)
        self.cleaned_data['links'] = self.cleaned_data[self.post_column].apply(
            self.extractor.find_urls)

        # Keep only posts without links
        self.cleaned_data = self.cleaned_data[self.cleaned_data['links'].str.len(
        ) == 0]

        removed_posts = original_size - len(self.cleaned_data)

        print(f"Number of posts removed that contained links: {removed_posts}")
        return self.cleaned_data

    def remove_duplicates(self):
        """Removes duplicate posts."""
        original_size = len(self.cleaned_data)
        # Remove duplicates, keeping only the first occurrence
        self.cleaned_data = self.cleaned_data.drop_duplicates(
            subset=self.post_column, keep='first')
        removed_duplicates = original_size - len(self.cleaned_data)
        print(f"Number of duplicate posts removed: {removed_duplicates}")
        return self.cleaned_data

    def filter_by_length(self):
        """Filters posts by their length (min_length and max_length)."""
        original_size = len(self.cleaned_data)
        # Add a column that contains the length of each post
        self.cleaned_data['post_length'] = self.cleaned_data[self.post_column].apply(
            len)
        # Filter posts based on the specified minimum and maximum length
        self.cleaned_data = self.cleaned_data[
            (self.cleaned_data['post_length'] >= self.min_length) &
            (self.cleaned_data['post_length'] <= self.max_length)
        ]

        filtered_size = len(self.cleaned_data)
        removed_posts = original_size - filtered_size
        print(f"Number of posts removed by length filter: {removed_posts}")
        return self.cleaned_data

    def preprocess(self, steps=None):
        """
        steps (list of str): List of preprocessing steps to execute. 
                             Possible values: ['remove_empty_posts', 'remove_posts_with_links', 
                             'remove_duplicates', 'filter_by_length'].
                             If None, all steps will be executed by default.
        """
        if steps is None:
            steps = ['remove_empty_posts', 'remove_posts_with_links',
                     'remove_duplicates', 'filter_by_length']

        for step in steps:
            if hasattr(self, step):
                method = getattr(self, step)
                method()  # Execute the method
            else:
                print(f"Warning: Step '{
                      step}' does not exist in the PostProcessor.")

        return self.cleaned_data
