import matplotlib.pyplot as plt
from sklearn import datasets
import os, collections
from eval_clustering import eval_clustering

# Load the iris dataset
iris = datasets.load_iris()

# Split the data into features and labels
X = iris.data
y = iris.target

range_clusters = list(range(2,10+1))
labelled_datapoints, \
   nongt_metrics, \
      gt_metrics = eval_clustering(X=X, gt_labels=y,
                                    algorithm='k-Means',
                                    num_clusters=range_clusters, num_runs=10,
                                    annotate_topN_best_scores=True, annotN=3,
                                    make_metrics_plots=True,
                                    make_silhoutte_plots=False,
                                    show=True, 
                                    save=True, RESULTS_DIR=os.getcwd()+'/results',
                                )