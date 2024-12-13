from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np


def calculate_average_cosine_similarity(data, cluster_column, embeddings):
    """
    Calculates the average cosine similarity within each cluster.

    Parameters:
    - data (pd.DataFrame): DataFrame containing cluster data.
    - cluster_column (str): Name of the column with cluster numbers.
    - embeddings (np.ndarray): Array of embeddings where rows correspond to rows in `data`.

    Returns:
    - dict: A dictionary where the key is the cluster number, and the value is the average cosine similarity.
    """

    unique_clusters = data[cluster_column].unique()

    average_cosine_similarities = {}

    for cluster in unique_clusters:

        cluster_indices = data[data[cluster_column] == cluster].index
        cluster_embeddings = embeddings[cluster_indices]

        if len(cluster_embeddings) > 1:

            cosine_sim = cosine_similarity(cluster_embeddings)
            # Remove diagonal values (ones – similarity of a vector with itself)
            mean_similarity = np.mean(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
            average_cosine_similarities[cluster] = mean_similarity
        else:
            # If the cluster has only one point, similarity cannot be computed
            average_cosine_similarities[cluster] = None

    return average_cosine_similarities


def get_cluster_problems(data, cluster_column, cluster_number, limit=None):
    """
    Retrieves a list of problems from a specified cluster.

    Parameters:
    - data (pd.DataFrame): The dataset containing the clusters and problems.
    - cluster_column (str): The name of the column containing cluster numbers.
    - cluster_number (int): The specific cluster number to retrieve problems from.
    - limit (int, optional): The maximum number of problems to display (default: None, meaning all problems).
    """

    problems_in_cluster = data[data[cluster_column] == cluster_number][
        "problems"
    ].tolist()

    if limit:
        problems_in_cluster = problems_in_cluster[:limit]

    print(f"Problems from the cluster {cluster_number}:")
    for problem in problems_in_cluster:
        print(f"- {problem}")

    return problems_in_cluster


def plot_cluster_distributions(data, cluster_columns, palette="viridis"):
    """
    Plots the distributions of clusters for given clustering methods.

    Parameters:
    - data (pd.DataFrame): Dataset containing cluster information.
    - cluster_columns (list of str): List of column names for clustering methods.
    - palette (str, optional): Color palette for the plots (default: "viridis").
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, column in enumerate(cluster_columns):
        ax = axes[i]
        cluster_counts = data[column].value_counts().sort_index()

        sns.barplot(
            x=cluster_counts.index,
            y=cluster_counts.values,
            hue=cluster_counts.index,
            palette=palette,
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Point Distribution by Clusters ({column})", fontsize=14)
        ax.set_xlabel("Cluster Number", fontsize=12)
        ax.set_ylabel("Number of Points", fontsize=12)

        x_ticks = np.arange(
            0, len(cluster_counts.index), max(1, len(cluster_counts.index) // 10)
        )
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(cluster_counts.index[x_ticks], rotation=45)

    plt.tight_layout()
    plt.show()


def detect_and_label_outliers(
    data, cluster_column, embeddings, threshold=0.7, outlier_label=-1
):
    """
    Assigns outlier labels to elements in clusters based on cosine similarity to the cluster centroid.

    Parameters:
    - data (pd.DataFrame): The dataset containing cluster information.
    - cluster_column (str): The name of the column containing cluster labels.
    - embeddings (np.ndarray): Array of embeddings where rows correspond to rows in `data`.
    - threshold (float): The minimum cosine similarity to consider an element as part of a cluster.
    - outlier_label (int or str): The label to assign to outliers (default: -1).

    Returns:
    - pd.DataFrame: Updated dataset with outlier labels assigned.
    """
    updated_data = data.copy()

    unique_clusters = updated_data[cluster_column].unique()

    centroids = {}
    for cluster in unique_clusters:
        cluster_indices = updated_data[updated_data[cluster_column] == cluster].index
        cluster_embeddings = embeddings[cluster_indices]
        centroids[cluster] = np.mean(cluster_embeddings, axis=0)

    for cluster in unique_clusters:
        cluster_indices = updated_data[updated_data[cluster_column] == cluster].index
        cluster_embeddings = embeddings[cluster_indices]

        centroid = centroids[cluster].reshape(1, -1)
        similarities = cosine_similarity(cluster_embeddings, centroid).flatten()

        # Update cluster labels for elements below the threshold
        for idx, similarity in zip(cluster_indices, similarities):
            if similarity < threshold:
                # Use specified outlier label
                updated_data.loc[idx, cluster_column] = outlier_label

    return updated_data
