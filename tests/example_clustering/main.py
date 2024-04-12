import os
import numpy as np
from sklearn.datasets import make_blobs, load_iris, load_digits
from eval_clustering import eval_clustering

## For testing purposes
rng = np.random.RandomState(0)
n_samples=1000

### Synthetic data: Without outliers
X, y = make_blobs(n_samples=n_samples, centers=5, n_features=2, cluster_std=0.60, random_state=rng)

# ### Synthetic data: With outliers
# centers = [[1, 1], [-1, -1], [1, -1]]
# #### Generate some blobs
# X, y = make_blobs(
#     n_samples=n_samples, centers=centers, cluster_std=0.4, random_state=rng
# )
# #### Change the first 1% entries to outliers
# for f in range(int(n_samples / 100)):
#     X[f] = [10, 3] + rng.normal(size=2) * 0.1
# #### Shuffle the data so that we don't know where the outlier is.
# X = shuffle(X, random_state=rng)

### Real benchmark
# data = load_iris()
# X, y = data.data, data.target

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

best_model, grid_search_results = eval_clustering(
                                       X=X,                                               # dataset to cluster
                                       gt_labels=y,                                       # ground-truth labels; often these aren't available so don't pass this argument
                                       num_runs=10,                                       # number of times to fit a model
                                       best_model_metric="FMI",                           # metric to use to choose the best model
                                       make_silhoutte_plots=True, embed_data_in_2d=False, # whether to make silhouette plots
                                       show=False,                                         # whether to display the plots; this is used in a notebook
                                       save=True, save_dir="results"                      # whether to save the plots
                                    )
print("Best model: ", best_model.__class__.__name__)
print("Best model params: ", best_model.get_params())