# Clustering Utilities

This packages provides a simple convenience wrapper around some basic sklearn utilities for clustering. The only function available is `eval_clustering()`.

## Installation
`pip install clustutils4r`

## Available Parameters

`model`: Clustering model object (untrained)

`X`: Numpy array containing preprocessed, normalized, complete dataset features

`gt_labels`: Numpy array containing encoded ground-truth labels for `X` (often not available)

`num_clusters`: Range of no. of clusters to grid search over.

`num_runs`: No. of runs per no. of cluster (defaults = 10).

`make_metrics_plots`: Plot various clustering evaluation metrics. Depending on whether `gt_labels` are provided, you will get one or two sets of plots - one for non-ground truth-based metrics and another for ground truth-based metrics. (default = `True`).

`annotate_topN_best_scores`: Whether to annotate top N score in metrics plots (defined by `annotN`) (defaults = `True`).

`annotN`: No. of top scores to annotated in metrics plots (defaults = 3).

`make_silhoutte_plots`: Whether to make silhouette plots for each `num_clusters` value (default = `False`).

`embed_data_in_2d`: Whether to compute TSNE embeddings of the `X` to plotted alongside silhouette plot or plot the first 2 features (default = `False`).

`RESULTS_DIR`: location to store results; directory will be created if it does not exist

`save`: set True if you want to save all results in RESULTS_DIR; defaults to False

`show`: display all results; useful in notebooks; defaults to False

## Example Usage
```python
import matplotlib.pyplot as plt
%matplotlib inline

import os, collections
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from clustutils4r.eval_clustering import eval_clustering

## Load the dataset
X, y = datasets.make_blobs(n_samples=5*100, 
                           centers=5, 
                           n_features=2, 
                           random_state=0)

## Setup model
model = KMeans()
model_params = {'init':'k-means++', 'n_init':1}
n_clusters_param_name='n_clusters'

## Run the evaluation
range_clusters = list(range(3,10+1))
labelled_datapoints, \
   nongt_metrics, \
      gt_metrics = eval_clustering(X=X, gt_labels=y,
                                    model=model, model_params=model_params, n_clusters_param_name=n_clusters_param_name,
                                    num_clusters=range_clusters, num_runs=10,
                                    annotate_topN_best_scores=True, annotN=3,
                                    make_metrics_plots=True,
                                    make_silhoutte_plots=False,
                                    show=True, 
                                    save=True, RESULTS_DIR=os.getcwd()+'/results',
                                )

```

![ch](tests/example_clustering/results/nongt_metrics_plots/cal_har.png)
![sil](tests/example_clustering/results/nongt_metrics_plots/dav_bou.png)
![sil](tests/example_clustering/results/nongt_metrics_plots/hopkn.png)
![sil](tests/example_clustering/results/nongt_metrics_plots/sil.png)

<!-- ![ch](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/nongt_metrics_plots/cal_har.png)
![sil](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/nongt_metrics_plots/dav_bou.png)
![sil](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/nongt_metrics_plots/hopkn.png)
![sil](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/nongt_metrics_plots/sil.png) -->

![fmi](tests/example_clustering/results/gt_metrics_plots/fmi.png)
![mi](tests/example_clustering/results/gt_metrics_plots/mutual_info.png)
![hcv](tests/example_clustering/results/gt_metrics_plots/hcv.png)
![ri](tests/example_clustering/results/gt_metrics_plots/rand_index.png)

<!-- ![fmi](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/gt_metrics_plots/fmi.png)
![mi](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/gt_metrics_plots/mutual_info.png)
![hcv](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/gt_metrics_plots/hcv.png)
![ri](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/gt_metrics_plots/rand_index.png) -->

![silplt](tests/example_clustering/results/silhouette_plots/5_silhouette_plot.png)

<!-- ![silplt](https://github.com/rutujagurav/clustutils4r/blob/main/tests/example_clustering/results/silhouette_plots/10_silhouette_plot.png) -->