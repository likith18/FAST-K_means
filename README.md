# Industry Clustering Model on Google Colab

This notebook provides an industry clustering model using machine learning techniques to categorize industries into optimal clusters and provide recommendations for related industries. The model leverages **Sentence Transformers** for embeddings, **KMeans** clustering, and **various clustering metrics** to evaluate the best number of clusters.

## Features
- **Clustering Optimization**: Finds the optimal number of clusters using metrics like silhouette score, Davies-Bouldin score, Calinski-Harabasz score, and inertia.
- **Recommendation System**: Provides related industry recommendations based on the trained clusters.
- **Dimensionality Reduction Visualization**: Uses UMAP to visualize clusters.


## Usage

### Step 1: Prepare the Data
Ensure your data file `new_structured_data.json` is in the same directory. This JSON file should contain industry data with main and sub-industries.

### Step 2: Run the Code
Execute the code cells in sequence to:

Load and preprocess the industry data.
Fit the model to find optimal clusters and assign each industry to a cluster.
Visualize the metrics and clusters.
Save the clustering results to improved_linkedin_industry_clusters.json.
This will:
- Load and preprocess the industry data.
- Fit the model to find optimal clusters and assign each industry to a cluster.
- Visualize the metrics and clusters.
- Save the clustering results to `improved_linkedin_industry_clusters.json`.

### Step 3: Get Industry Recommendations
Use `get_industry_recommendations` function to retrieve industry recommendations for a list of inputs.

Example:
```python
input_industries = [
    "Health Tech Innovator",
    "Retail Merchandising Specialist",
    "Brand Strategy Consultant"
]

industry_recommendations = get_industry_recommendations(input_industries)
print(industry_recommendations)
```

This function outputs the main industry recommended for each input industry.

## Configuration
Adjust the model parameters by modifying `ClusteringConfig` in `main.py`. This allows you to set:
- The range for the number of clusters (`min_clusters` and `max_clusters`).
- Model name for embeddings (`model_name`).
- Batch size and device for computation.
