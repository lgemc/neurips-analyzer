# NeurIPS analyzer

## Why it exists

I was wondering, how can I be up to date with the latest trends and most important papers in 
machine learning research? various conferences and indexed journals exists, but it's hard to 
keep track of all of them, or to know which ones are more relevant in the field.

NeurIPS analyzer allows you to respond to two main questions:

1. What are the most important 7 groups of papers by year?
2. What is the evolution of topics over the years?

It also allows you to search for specific keywords in the papers.

## Methodology

In order to generate the paper groups, NeurIPS papers were downloaded, including years from 2010 to 2025, from 
the oficial NeurIPS [website](https://neurips.cc/Downloads/2010). 

Then, the `all-MiniLM-L6-v2` embedding model from the `sentence-transformers` library is used to generate
embeddings for each paper abstract. As you can see on hugging face embed [benchmarks](https://huggingface.co/spaces/mteb/leaderboard) 
this model has a good trade-off between model size and accuracy.

After that, the KMeans clustering algorithm is used to group the papers into 7 clusters. The number 7
was chosen after some experimentation with `elbow method` and `silhouette score`, where 4, 5 and 7 
clusters had the maximum scores, cluster number from 3 to 15 were tested. Even if the maximum 
score was for 4 clusters, I found that 7 clusters gave more interpretable results.

## Pipeline 

As you can see in `pipeline/run.py` it orchestrates the whole process, assuming than 
the papers are already downloaded in the `paper_list` folder.

It executes the following steps:

1. Load the papers from the `paper_list` folder and store them in `sqlite` database.
2. Generate embeddings for each paper abstract and store them in the database using `GaussianMixture` to allow soft clustering.
3. For each year from 2010 to 2025, cluster the papers into 7 clusters using KMeans.
4. For each cluster, get the first 20 most relevant papers to the cluster center and generate a cluster name and description.
5. Generate umap visualizations for each year, and store them in sqlite database.

## How it looks

![main_visualization](static/main_view.png)