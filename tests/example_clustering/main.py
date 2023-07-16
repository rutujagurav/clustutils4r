import matplotlib.pyplot as plt
import os, collections
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch import mode

from eval_clustering import eval_clustering

## Load the iris dataset
# data = datasets.load_digits()
# X = data.data
# y = data.target

X, y = datasets.make_blobs(n_samples=5*100, 
                           centers=5, 
                           n_features=2, 
                           random_state=0)

## Setup model
model = KMeans()
model_params = {'init':'k-means++', 'n_init':1}
n_clusters_param_name='n_clusters'

# model = GaussianMixture()
# model_params = {'covariance_type':'full', 'n_init':1}
# n_clusters_param_name='n_components'

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