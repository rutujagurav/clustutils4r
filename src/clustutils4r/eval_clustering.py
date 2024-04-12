import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import sklearn
print(f"sklearn version: {sklearn.__version__}")
# from sklearn.externals.joblib import parallel_backend

from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.manifold import TSNE
# from openTSNE import TSNE
from sklearn.decomposition import PCA

from sklearn.datasets import make_blobs, load_iris, load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

## Features Clustering
from sklearn.cluster import KMeans, SpectralClustering, \
    AgglomerativeClustering, \
    AffinityPropagation, MeanShift, \
    DBSCAN, HDBSCAN, \
    OPTICS, Birch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import make_scorer

# from skopt import BayesSearchCV

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import collections, os, sys, random, functools, pdb, joblib, json, inspect
from pprint import pprint
from tqdm.autonotebook import tqdm

## Hopkins Statistic
from sklearn.neighbors import NearestNeighbors
# from kneed import KneeLocator

def hopkins_statistic_v2(X, frac=0.5, seed=42):
  n = X.shape[0]
  d = X.shape[1]
  m = int(frac * n) 

  np.random.seed(seed)
  nbrs = NearestNeighbors(n_neighbors=1).fit(X)
  # u_dist
  rand_X = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(m,d))
  u_dist = nbrs.kneighbors(rand_X, return_distance=True)[0]
  # w_dist
  idx = np.random.choice(n, size=m, replace=False)
  w_dist = nbrs.kneighbors(X[idx,:], 2, return_distance=True)[0][:,1]

  print(u_dist.shape, w_dist.shape, d)
  U = np.power(u_dist, d).sum() #(u_dist**d).sum()
  W = np.power(w_dist, d).sum() #(w_dist**d).sum()
  H = U / (U + W)
  return H

# from pyclustertend import hopkins
# def hopkins_statistic(X, frac=0.5):
#     return hopkins(X, int(frac*len(X)))

# def dbscan_eps_guess(X):
#     '''
#     Reference- https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan
#     '''
#     ## Choice of minPts = 2 * number of features in the dataset. Note: This is an arbitrary choice.
#     minPts = 2*X.shape[1]

#     ## Choice of eps based on k-Nearest Neighbors
#     neighbors = NearestNeighbors(n_neighbors=minPts).fit(X)
#     distances, indices = neighbors.kneighbors(X)
#     print(distances.shape, indices.shape)
#     distances = sorted(distances[:,minPts-1], reverse=True)

#     kneeloc = KneeLocator(range(1, len(distances)+1), distances, curve='convex', direction='decreasing')
#     # kneeloc.plot_knee_normalized()
#     eps = np.round(kneeloc.knee_y, 2)

#     return eps, minPts

def calc_gt_metrics(gt_labels=[], cluster_labels=[]):
    metric_names = [
                    'RI', 
                    'ARI', 
                    'MI', 
                    'AMI', 'NMI', 
                    'Homogeneity', 'Completeness', 
                    'V-Measure', 
                    'FMI'
                ]
    ri = np.round(rand_score(gt_labels, cluster_labels), 3)
    ari = np.round(adjusted_rand_score(gt_labels, cluster_labels),3)
    
    mi = mutual_info_score(gt_labels, cluster_labels).round(3)
    ami = adjusted_mutual_info_score(gt_labels, cluster_labels).round(3)
    nmi = normalized_mutual_info_score(gt_labels, cluster_labels).round(3)

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(gt_labels, cluster_labels).round(3)

    fmi = fowlkes_mallows_score(gt_labels, cluster_labels).round(3)

    return dict(zip(metric_names, [ri, ari, mi, ami, nmi, homogeneity, completeness, v_measure, fmi]))

def calc_nongt_metrics(X=None, cluster_labels=None):
    metric_names = ['Silhouette', 
                                'Calinski-Harabasz', 'Davies-Bouldin', 
                                # 'Hopkins'
                            ]
    if len(set(cluster_labels)) == 1:
        print("Only 1 cluster found. Skipping metrics calculation...")
        silhouette_avg = 0.0
        calinski_harabasz_score_avg = 0.0
        davies_bouldin_score_avg = 0.0
    else:
        silhouette_avg = silhouette_score(X, cluster_labels).round(3)
        calinski_harabasz_score_avg = calinski_harabasz_score(X, cluster_labels).round(3)
        davies_bouldin_score_avg = davies_bouldin_score(X, cluster_labels).round(3)
    
    # return silhouette_avg,calinski_harabasz_score_avg, davies_bouldin_score_avg
    return dict(zip(metric_names, [silhouette_avg, 
                                    calinski_harabasz_score_avg, 
                                    davies_bouldin_score_avg, 
                                ]))

def plot_silhouette_analysis(X=None, cluster_labels=None,
                             titlestr="Silhouette Analysis",
                             embed_data_in_2d=False,
                             show=False, save=False, save_dir=None):
    print("Plotting silhouette analysis...")
    
    n_clusters = len(np.unique(cluster_labels))
    ## Create a subplot with 1 row and 2 columns
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    # fig.set_size_inches(18, 7)
    '''
    The 1st subplot is the silhouette plot
    The silhouette coefficient can range from -1, 1 but in this example all
    lie within [-0.1, 1] so maybe ax1.set_xlim([-0.1, 1])?
    The (n_clusters+1)*10 is for inserting blank space between silhouette
    plots of individual clusters, to demarcate them clearly.
    '''
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    ## Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    ## Compute the avg silhouette score for the whole dataset
    silhouette_avg = silhouette_score(X, cluster_labels).round(3)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=20)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters", fontsize=20)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=20)
    ax1.set_ylabel("Cluster label", fontsize=20)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Avg. Silhouette Score")
    # ax1.text(silhouette_avg+0.05, 100.0, "avg. silhouette score = {}".format(silhouette_avg), color="red", fontsize=20, rotation=90)

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    if embed_data_in_2d:
        print("Embedding data in 2-D...")
        X_embedded = TSNE(n_components=2).fit_transform(X)
        # X_embedded = Isomap(n_components=2).fit_transform(X)
        # X_embedded = SpectralEmbedding(n_components=2).fit_transform(X)
        # X_embedded = PCA(n_components=2).fit_transform(X)
        ax2.scatter(
            X_embedded[:, 0], X_embedded[:, 1], marker=".", s=100, lw=0, alpha=1, c=colors, edgecolor="k"
        )
        ax2.set_xlabel("TSNE axis-1")
        ax2.set_ylabel("TSNE axis-2")
        
    else:
        print("Plotting first 2 features of the data...")
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=100, lw=0, alpha=1, c=colors, edgecolor="k"
        )

    # # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # # Draw white circles at cluster centers
    # ax2.scatter(
    #     centers[:, 0],
    #     centers[:, 1],
    #     marker="o",
    #     c="white",
    #     alpha=1,
    #     s=200,
    #     edgecolor="k",
    # )
    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data", fontsize=20)
    # ax2.set_xlabel("TSNE axis-1")
    # ax2.set_ylabel("TSNE axis-2")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")
    ax2.axis("off")
    ax2.grid(False)

    plt.suptitle(
        f"{titlestr} with n_clusters = {n_clusters}",
        fontsize=20,
        fontweight="bold",
    )
    
    plt.tight_layout()
    
    if save:
        save_dir_sil = f"{save_dir}/silhouette_plots"
        os.makedirs(save_dir_sil, exist_ok=True)
        print("Saving silhouette plot for k={} at {}"\
                .format(n_clusters, save_dir_sil+f'/{n_clusters}_silhouette_plot.png'))
        plt.savefig(save_dir_sil+f'/{n_clusters}_silhouette_plot.png', dpi=300)
    if show:
        plt.show()
    plt.close()

    # fig, ax = plt.subplots(1, 1, figsize=(10,10))
    # if not embed_data_in_2d:
    #     ax.scatter(X[:, 0], X[:, 1], marker=".", s=100, lw=0, alpha=1, c='grey', edgecolor='grey')
    # else:
    #     ax.scatter(X_embedded[:, 0], X_embedded[:, 1], marker=".", s=100, lw=0, alpha=1, c='grey', edgecolor='grey')
    # ax.axis('off')
    # ax.grid(False)
    # plt.savefig(os.path.join(save_dir_sil, 'data_viz_plot.png'), dpi=300)
    # plt.close()

def plot_grid_search_metrics(hparam_search_results, metric="Calinski-Harabasz", show=False, save=False, save_dir=None):
    hparams = [col for col in hparam_search_results.columns if col.startswith("param_")]
    metrics = [col for col in hparam_search_results.columns if col.startswith("mean_test_")]

    dimslist4parcoordplot = []
    for hparam_name in hparams:
        if hparam_search_results[hparam_name].dtype in ["object", "categorical"]: 
            le = LabelEncoder()
            encoded_hparam = le.fit_transform(hparam_search_results[hparam_name])
            dimslist4parcoordplot.append(
                dict(   
                        ticktext=le.classes_, tickvals=le.transform(le.classes_),
                        label=hparam_name.split("param_")[-1], 
                        values=encoded_hparam
                    )
            )
        else:
            dimslist4parcoordplot.append(
                dict(
                        label=hparam_name.split("param_")[-1], 
                        values=hparam_search_results[hparam_name].values
                    )
            )
    for metric_name in metrics:
        dimslist4parcoordplot.append(
            dict(
                    label=metric_name.split("mean_test_")[-1], 
                    values=hparam_search_results[metric_name].values
                )
        )

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = hparam_search_results[f'mean_test_{metric}'],
                    colorscale = 'Viridis',
                    showscale = True,
                ),
            dimensions = dimslist4parcoordplot
        )
    )
    # fig.update_layout(width=1500, height=500)
    if save:
        fig.write_html(save_dir+"/parcoord_plot.html")
        fig.write_image(save_dir+"/parcoord_plot.png")
    if show:
        fig.show()

def cluster_feats(X=None, gt_labels=[],
                n_clusters_range = range(3,21),
                clustering_algorithms=["KMeans", "AgglomerativeClustering", "HDBSCAN", "MeanShift"], 
                num_runs=1, best_model_metric="Calinski-Harabasz",
                make_silhoutte_plots=False, embed_data_in_2d = False,
                show=False, save=False, save_dir=None):
    '''
    Cluster fixed-length segments of time as represented by a set of statistical features derived from time-segments data of all channels using partition clustering approach.
    In partition clustering approach, the user must provide the number of clusters as a parameter and the dataset gets partitioned into those many clusters.
    '''
    # if len(gt_labels)!=0:
    #     print("Ground truth labels provided. Computing ground truth-based clustering metrics along with non-ground truth based metrics.") 
    
    # print("[Implementation: pyclustertend] Hopkins Statistic (0=clustered, 0.5=randomly distributed) : {}".format(hopkins_statistic(X).round(3)))
    # print("[Implementation: mine] Hopkins Statistic (1=clustered, 0.5=random, 0=uniforrm) : {}".format(hopkins_statistic_v2(X).round(3)))
   
    if len(gt_labels)!=0:
        print("Ground truth labels provided. Computing ground truth-based clustering metrics along with non-ground truth based metrics.")
        def scorer(estimator, X, y):
            estimator.fit(X)
            cluster_labels = estimator.labels_
            num_labels = len(set(cluster_labels))
            num_samples = len(X)
            if num_labels == 1 or num_labels == num_samples:
                return {
                        "Silhouette": -1,
                        "Calinski-Harabasz": float('-inf'),
                        "Davies-Bouldin": 1,
                        "RI": 0,
                        "ARI": -1,
                        "MI": 0,
                        "AMI": 0,
                        "NMI": 0,
                        "Homogeneity": 0,
                        "Completeness": 0,
                        "V-Measure": 0,
                        "FMI": 0
                    }
            else:
                homogeneity_score, completeness_score, v_measure_score = homogeneity_completeness_v_measure(y, cluster_labels)
                return {"Silhouette": silhouette_score(X, cluster_labels),
                        "Calinski-Harabasz": calinski_harabasz_score(X, cluster_labels),
                        "Davies-Bouldin": davies_bouldin_score(X, cluster_labels),
                        "RI": rand_score(y, cluster_labels),
                        "ARI": adjusted_rand_score(y, cluster_labels),
                        "MI": mutual_info_score(y, cluster_labels),
                        "AMI": adjusted_mutual_info_score(y, cluster_labels),
                        "NMI": normalized_mutual_info_score(y, cluster_labels),
                        "Homogeneity": homogeneity_score,
                        "Completeness": completeness_score,
                        "V-Measure": v_measure_score,
                        "FMI": fowlkes_mallows_score(y, cluster_labels)
                    }
    else:
        print("Ground truth labels not provided. Computing non-ground truth based clustering metrics only.")
        # scorer = make_scorer(calinski_harabasz_score)
        def scorer(estimator, X):
            estimator.fit(X)
            cluster_labels = estimator.labels_
            num_labels = len(set(cluster_labels))
            num_samples = len(X)
            if num_labels == 1 or num_labels == num_samples:
                return {
                        "Silhouette": -1,
                        "Calinski-Harabasz": -1e9,
                        "Davies-Bouldin": 1
                    }
            else:
                return {"Silhouette": silhouette_score(X, cluster_labels),
                        "Calinski-Harabasz": calinski_harabasz_score(X, cluster_labels),
                        "Davies-Bouldin": davies_bouldin_score(X, cluster_labels)
                    }

    search_space = []            
    if "KMeans" in clustering_algorithms:
        search_space.append([KMeans(), {
                                        "n_clusters": n_clusters_range, 
                                        "algorithm": ["lloyd", "elkan"],
                                        "init": ["k-means++", "random"], 
                                        "max_iter": [50, 100, 200], 
                                        "tol": [1e-2, 1e-4, 1e-6, 1e-8],
                                    }])
    if "SpectralClustering" in clustering_algorithms:
        search_space.append([SpectralClustering(), {
                                                    "n_clusters": n_clusters_range,
                                                    "affinity": ["nearest_neighbors", "rbf", "polynomial"],
                                                    "n_neighbors": [int(0.01*len(X)), int(0.05*len(X)), int(0.1*len(X))],
                                                    "degree": [3, 4, 5],
                                                    "assign_labels": ["kmeans", "discretize"],
                                                }])

    if "AgglomerativeClustering" in clustering_algorithms:
        search_space.append([AgglomerativeClustering(), {
                                                        "n_clusters": n_clusters_range,
                                                        "metric": ["euclidean", "manhattan", "cosine", "l1", "l2"],
                                                        "linkage": ["ward", "complete", "average", "single"]
                                                    }])
    
    if "AffinityPropagation" in clustering_algorithms:
        search_space.append([AffinityPropagation(), {
                                                    "damping": [0.5, 0.6, 0.7, 0.8, 0.9],
                                                    "max_iter": [50, 100, 200],
                                                    "convergence_iter": [15, 20, 25]
                                                }])

    # if "DBSCAN" in clustering_algorithms:
    #     eps, minPts = dbscan_eps_guess(X)
    #     eps_choices, min_samples_choices = np.round([eps-0.1, eps-0.05, eps, eps+0.05,  eps+0.1], 2), [minPts]*5
    #     ### remove negative values
    #     neg_val_idx = np.argwhere(np.array(eps_choices)<=0).flatten()
    #     eps_choices, min_samples_choices = np.delete(eps_choices, neg_val_idx), np.delete(min_samples_choices, neg_val_idx)
    #     # eps_choices, min_samples_choices = np.round([eps-4, eps-2, eps, eps+2,  eps+4], 2), [50]*5
    #     print(f"DBSCAN: eps_choices={eps_choices}, min_samples_choices={min_samples_choices}")

    #     search_space.append([DBSCAN(), {
    #                                     "eps": eps_choices, 
    #                                     "min_samples": min_samples_choices
    #                                 }])

    if "HDBSCAN" in clustering_algorithms:
        # min_cluster_size = [int(0.01*len(X)), int(0.05*len(X)), int(0.1*len(X))]
        min_cluster_size = range(1,21)
        # print(f"HDBSCAN: min_cluster_size={min_cluster_size}")
        search_space.append([HDBSCAN(), {
                                        "min_cluster_size": min_cluster_size,
                                        "cluster_selection_method": ["eom", "leaf"],
                                        "allow_single_cluster": [True, False]
                                    }])
    if "MeanShift" in clustering_algorithms:
        search_space.append([MeanShift(), {
                                        "max_iter": [100, 300, 500],
                                        "bin_seeding": [True, False],
                                        "min_bin_freq": [1, 5, 10]
                                    }])

    print("Search space:")
    pprint(search_space)
    print()
    # sys.exit()
    
    grid_search_results = {}
    best_score_overall = float('-inf')  # Initialize best score to negative infinity
    best_estimator_overall = None
    for estimator, param_grid in tqdm(search_space):
        estimator_name = str(estimator.__class__.__name__)
        print(f"Searching for best hyperparameters for {estimator_name}...")
        print(f"Available parameters: {list(estimator.get_params().keys())}")
        print(f"But only searching for parameters: {list(param_grid.keys())}")

        if save:
            save_dir_ = os.path.join(save_dir, "models", estimator_name)
            if not os.path.exists(save_dir_): os.makedirs(save_dir_)
        else:
            save_dir_ = None

        cv = [(slice(None), slice(None)) for _ in range(num_runs)] # Ref- https://stackoverflow.com/a/44682305
        clusterer_hpopt = GridSearchCV(estimator=estimator, 
                                        param_grid=param_grid,
                                        scoring=scorer, 
                                        refit=best_model_metric,
                                        cv=cv,
                                        n_jobs=-1, 
                                        verbose=0)
        if len(gt_labels)!=0:
            clusterer_hpopt.fit(X, gt_labels)
        else:
            clusterer_hpopt.fit(X)

        best_model = clusterer_hpopt.best_estimator_

        # Check if current score is better than best score and update if true
        if clusterer_hpopt.best_score_ > best_score_overall:
            best_score_overall = clusterer_hpopt.best_score_
            best_estimator_overall = best_model
        
        grid_search_results_df = pd.DataFrame(clusterer_hpopt.cv_results_)
        grid_search_results[estimator_name] = grid_search_results_df
        ## Plot grid search results
        plot_grid_search_metrics(grid_search_results_df, 
                                 metric=best_model_metric,
                                 show=show, save=save, save_dir=save_dir_)

        ## Plot Silhouette analysis
        if make_silhoutte_plots:
            plot_silhouette_analysis(X=X, cluster_labels=best_model.labels_,
                                        embed_data_in_2d=embed_data_in_2d,
                                        show=show, save=save, save_dir=save_dir_)

        if save:
            ## Save cv_results_ to a json file
            grid_search_results_df.to_json(os.path.join(save_dir_, 'grid_search_results.json'))
            grid_search_results_df.to_csv(os.path.join(save_dir_, 'grid_search_results.csv'), index=False)

            ## Save best model params
            best_model_info = {"model_name":estimator_name, "model_params": best_model.get_params()}
            print()
            with open(os.path.join(save_dir_, 'best_model_info.json'), 'w') as f:
                json.dump(best_model_info, f, default=str)

            ## Save the best models using joblib
            joblib.dump(best_model, os.path.join(save_dir_, f'best_model__{estimator_name}.joblib'))
    
    if save:
        ## Save best overall model params
        best_model_info = {"model_name":best_estimator_overall.__class__.__name__, "model_params": best_estimator_overall.get_params()}
        with open(os.path.join(save_dir, 'best_model_info.json'), 'w') as f:
            json.dump(best_model_info, f, default=str)
        ## Save the best overall model using joblib
        joblib.dump(best_estimator_overall, os.path.join(save_dir, f'best_model.joblib'))

    return best_estimator_overall, grid_search_results

def eval_clustering(X=None, gt_labels=[], 
                    num_runs=1, 
                    best_model_metric="Calinski-Harabasz",
                    make_silhoutte_plots=False, embed_data_in_2d=False,
                    show=False, save=False, save_dir=None):
    
    return cluster_feats(
                            X=X, gt_labels=gt_labels,
                            num_runs=num_runs, best_model_metric=best_model_metric,
                            make_silhoutte_plots=make_silhoutte_plots, embed_data_in_2d=embed_data_in_2d,
                            show=show, save=save, save_dir=save_dir
                        )
    
if __name__ == "__main__":
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

    best_model = cluster_feats(
                                X=X, gt_labels=y,
                                num_runs=10,
                                make_silhoutte_plots=True, embed_data_in_2d=False,
                                show=False, save=True, save_dir=save_dir,
                            )
    print("Best model: ", best_model.__class__.__name__)
    print("Best model params: ", best_model.get_params())