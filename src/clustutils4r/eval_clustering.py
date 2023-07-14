import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.manifold import Isomap, SpectralEmbedding
# from sklearn.manifold import TSNE
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import fowlkes_mallows_score

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import collections, os, sys, random

from pyclustertend import hopkins
def hopkins_statistic(X, frac=0.5):
    return 1-hopkins(X, int(frac*len(X)))

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
    ri = rand_score(gt_labels, cluster_labels).round(3)
    ari = adjusted_rand_score(gt_labels, cluster_labels).round(3)
    
    mi = mutual_info_score(gt_labels, cluster_labels).round(3)
    ami = adjusted_mutual_info_score(gt_labels, cluster_labels).round(3)
    nmi = normalized_mutual_info_score(gt_labels, cluster_labels).round(3)

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(gt_labels, cluster_labels)

    fmi = fowlkes_mallows_score(gt_labels, cluster_labels).round(3)

    return dict(zip(metric_names, [ri, ari, mi, ami, nmi, homogeneity, completeness, v_measure, fmi]))

def calc_nongt_metrics(X=None, cluster_labels=None):
    metric_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Hopkins']
    silhouette_avg = silhouette_score(X, cluster_labels).round(3)
    calinski_harabasz_score_avg = calinski_harabasz_score(X, cluster_labels).round(3)
    davies_bouldin_score_avg = davies_bouldin_score(X, cluster_labels).round(3)
    hopkins_score = hopkins_statistic(X).round(3)
    
    # return silhouette_avg,calinski_harabasz_score_avg, davies_bouldin_score_avg, hopkins_score
    return dict(zip(metric_names, [silhouette_avg, calinski_harabasz_score_avg, davies_bouldin_score_avg, hopkins_score]))

def plot_gt_metrics(gt_metrics=None, num_clusters=None,
                    annotate_topN_best_scores=True, annotN=3,
                    show=False, save=False, save_dir=None):

    if isinstance(gt_metrics, dict):
        ri_scores_mean = gt_metrics['mean']['RI'].values
        ari_scores_mean = gt_metrics['mean']['ARI'].values
        mi_scores_mean = gt_metrics['mean']['MI'].values
        ami_scores_mean = gt_metrics['mean']['AMI'].values
        nmi_scores_mean = gt_metrics['mean']['NMI'].values
        homogeneity_scores_mean = gt_metrics['mean']['Homogeneity'].values
        completeness_scores_mean = gt_metrics['mean']['Completeness'].values
        v_measure_scores_mean = gt_metrics['mean']['V-Measure'].values
        fmi_scores_mean = gt_metrics['mean']['FMI'].values

        ri_scores_std = gt_metrics['std']['RI'].values
        ari_scores_std = gt_metrics['std']['ARI'].values
        mi_scores_std = gt_metrics['std']['MI'].values
        ami_scores_std = gt_metrics['std']['AMI'].values
        nmi_scores_std = gt_metrics['std']['NMI'].values
        homogeneity_scores_std = gt_metrics['std']['Homogeneity'].values
        completeness_scores_std = gt_metrics['std']['Completeness'].values
        v_measure_scores_std = gt_metrics['std']['V-Measure'].values
        fmi_scores_std = gt_metrics['std']['FMI'].values
    
    elif isinstance(gt_metrics, pd.DataFrame):
        ri_scores_mean = gt_metrics['RI'].values
        ari_scores_mean = gt_metrics['ARI'].values
        mi_scores_mean = gt_metrics['MI'].values
        ami_scores_mean = gt_metrics['AMI'].values
        nmi_scores_mean = gt_metrics['NMI'].values
        homogeneity_scores_mean = gt_metrics['Homogeneity'].values
        completeness_scores_mean = gt_metrics['Completeness'].values
        v_measure_scores_mean = gt_metrics['V-Measure'].values
        fmi_scores_mean = gt_metrics['FMI'].values

        ri_scores_std = [0*0]*len(ri_scores_mean)
        ari_scores_std = [0*0]*len(ari_scores_mean)
        mi_scores_std = [0*0]*len(mi_scores_mean)
        ami_scores_std = [0*0]*len(ami_scores_mean)
        nmi_scores_std = [0*0]*len(nmi_scores_mean)
        homogeneity_scores_std = [0*0]*len(homogeneity_scores_mean)
        completeness_scores_std = [0*0]*len(completeness_scores_mean)
        v_measure_scores_std = [0*0]*len(v_measure_scores_mean)
        fmi_scores_std = [0*0]*len(fmi_scores_mean)
    
    # fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## RI
    ax.plot(num_clusters, ri_scores_mean, label='RI')
    ax.fill_between(num_clusters, 
                        ri_scores_mean - ri_scores_std, 
                        ri_scores_mean + ri_scores_std, 
                        alpha=0.2)
    # axs[0,0].set_title("Rand Index")
    # axs[0,0].set_xlabel("No. of Clusters")

    ## ARI
    ax.plot(num_clusters, ari_scores_mean, label='ARI')
    ax.fill_between(num_clusters,
                        ari_scores_mean - ari_scores_std,
                        ari_scores_mean + ari_scores_std,
                        alpha=0.2)
    ax.set_title("Rand Index")
    ax.set_xlabel("No. of Clusters")
    # axs[0,0].set_ylim(0,1)
    if annotate_topN_best_scores:
        top_k_indices = np.array(ari_scores_mean).argsort()[-annotN:][::-1]
        for i in top_k_indices:
            value = ari_scores_mean[i]
            ax.text(i+num_clusters[0], value, f'{num_clusters[i]}', color='red')

    ax.legend(title='Metric', loc='best')

    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'gt_metrics_plots/rand_index.png'), dpi=300)
    if show:
        plt.show()
    plt.close()    


    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## MI
    # axs[0,1].plot(num_clusters, mi_scores_mean, label='MI')
    # axs[0,1].fill_between(num_clusters,
    #                     mi_scores_mean - mi_scores_std,
    #                     mi_scores_mean + mi_scores_std,
    #                     alpha=0.2)

    ## AMI
    ax.plot(num_clusters, ami_scores_mean, label='AMI')
    ax.fill_between(num_clusters,
                        ami_scores_mean - ami_scores_std,
                        ami_scores_mean + ami_scores_std,
                        alpha=0.2)

    ## NMI
    ax.plot(num_clusters, nmi_scores_mean, label='NMI')
    ax.fill_between(num_clusters,
                        nmi_scores_mean - nmi_scores_std,
                        nmi_scores_mean + nmi_scores_std,
                        alpha=0.2)
    
    ax.set_xlabel("No. of Clusters")
    # axs[0,1].set_ylim(0,1)
    if annotate_topN_best_scores:
        top_k_indices = np.array(nmi_scores_mean).argsort()[-annotN:][::-1]
        for i in top_k_indices:
            value = nmi_scores_mean[i]
            ax.text(i+num_clusters[0], value, f'{num_clusters[i]}', color='red')
    ax.legend(title='Metric', loc='best')

    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'gt_metrics_plots/mutual_info.png'), dpi=300)
    if show:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## Homogeneity
    ax.plot(num_clusters, homogeneity_scores_mean, label='Homogeneity')
    ax.fill_between(num_clusters,
                        homogeneity_scores_mean - homogeneity_scores_std,
                        homogeneity_scores_mean + homogeneity_scores_std,
                        alpha=0.2)

    ## Completeness
    ax.plot(num_clusters, completeness_scores_mean, label='Completeness')
    ax.fill_between(num_clusters,
                        completeness_scores_mean - completeness_scores_std,
                        completeness_scores_mean + completeness_scores_std,
                        alpha=0.2)

    ## V-Measure
    ax.plot(num_clusters, v_measure_scores_mean, label='V-Measure')
    ax.fill_between(num_clusters,
                        v_measure_scores_mean - v_measure_scores_std,
                        v_measure_scores_mean + v_measure_scores_std,
                        alpha=0.2)
    
    ax.set_xlabel("No. of Clusters")
    # ax.set_ylim(0,1)
    if annotate_topN_best_scores:
        top_k_indices = np.array(v_measure_scores_mean).argsort()[-annotN:][::-1]
        for i in top_k_indices:
            value = v_measure_scores_mean[i]
            ax.text(i+num_clusters[0], value, f'{num_clusters[i]}', color='red')
    ax.legend(title='Metric', loc='best')

    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'gt_metrics_plots/hcv.png'), dpi=300)
    if show:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## FMI
    ax.plot(num_clusters, fmi_scores_mean, label='FMI')
    ax.fill_between(num_clusters,
                        fmi_scores_mean - fmi_scores_std,
                        fmi_scores_mean + fmi_scores_std,
                        alpha=0.2)
    ax.set_title("Fowlkes-Mallows Index")
    ax.set_xlabel("No. of Clusters")
    # ax.set_ylim(0,1)
    if annotate_topN_best_scores:
        top_k_indices = np.array(fmi_scores_mean).argsort()[-annotN:][::-1]
        for i in top_k_indices:
            value = fmi_scores_mean[i]
            ax.text(i+num_clusters[0], value, f'{num_clusters[i]}', color='red')
    # axs[1,1].legend(title='Metric', loc='best')
    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'gt_metrics_plots/fmi.png'), dpi=300)
    if show:
        plt.show()
    plt.close()


    # plt.tight_layout()    
    # if save:
    #     print("Saving ground truth-based clustering metrics plot at {}".format(os.path.join(save_dir, 'feats_clustering_metrics.png')))
    #     plt.savefig(os.path.join(save_dir, 'feats_clustering_gt_metrics.png'), dpi=300)
    # if show:
    #     plt.show()
    # plt.close()    


def plot_nongt_metrics(nongt_metrics=None, 
                       num_clusters=None, annotate_topN_best_scores=False, annotN=3,
                       show=False, save=False, save_dir=None):
    if isinstance(nongt_metrics, dict):
        sil_scores_mean = nongt_metrics['mean']['Silhouette'].values
        ch_scores_mean = nongt_metrics['mean']['Calinski-Harabasz'].values
        db_scores_mean = nongt_metrics['mean']['Davies-Bouldin'].values
        hpkn_scores_mean = nongt_metrics['mean']['Hopkins'].values

        sil_scores_std = nongt_metrics['std']['Silhouette'].values
        ch_scores_std = nongt_metrics['std']['Calinski-Harabasz'].values
        db_scores_std = nongt_metrics['std']['Davies-Bouldin'].values
        hpkn_scores_std = nongt_metrics['std']['Hopkins'].values

    elif isinstance(nongt_metrics, pd.DataFrame):
        sil_scores_mean = nongt_metrics['Silhouette'].values
        ch_scores_mean = nongt_metrics['Calinski-Harabasz'].values
        db_scores_mean = nongt_metrics['Davies-Bouldin'].values
        hpkn_scores_mean = nongt_metrics['Hopkins'].values

        sil_scores_std = [0.0]*len(sil_scores_mean)
        ch_scores_std = [0.0]*len(ch_scores_mean)
        db_scores_std = [0.0]*len(db_scores_mean)
        hpkn_scores_std = [0.0]*len(hpkn_scores_mean)
    
    # fig, axs = plt.subplots(2,2, figsize=(10,10))

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## Silhouette
    ax.plot(num_clusters, sil_scores_mean)
    ax.fill_between(num_clusters, 
                        sil_scores_mean - sil_scores_std, 
                        sil_scores_mean + sil_scores_std, 
                        color='b', alpha=0.2)
    ax.set_title("Avg. Silhoutte Score\n(-1=incorrect, 0=overlap, 1=dense)")
    ax.set_xlabel("No. of Clusters")
    ax.set_ylim(-1.1,1.1)

    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'nongt_metrics_plots/sil.png'), dpi=300)
    if show:
        plt.show()
    plt.close()
    
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## Calinski-Harabasz
    ax.plot(num_clusters, ch_scores_mean)
    ax.fill_between(num_clusters, 
                        ch_scores_mean - ch_scores_std, 
                        ch_scores_mean + ch_scores_std, 
                        color='b', alpha=0.2)
    ax.set_title("Calinski Harabasz Score\n(higher=dense,well-separated)")
    ax.set_xlabel("No. of Clusters")
    
    if annotate_topN_best_scores:
        # # top_k_indices = sorted(range(len(ch_scores)), key=lambda i: ch_scores[i], reverse=True)[:k]
        top_k_indices = np.array(ch_scores_mean).argsort()[-annotN:][::-1]
        for i in top_k_indices:
            value = ch_scores_mean[i]
            ax.text(i+num_clusters[0], value, f'{num_clusters[i]}', color='red')

    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'nongt_metrics_plots/cal_har.png'), dpi=300)
    if show:
        plt.show()
    plt.close()


    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## Davies-Bouldin
    ax.plot(num_clusters, db_scores_mean)
    ax.fill_between(num_clusters, 
                        db_scores_mean - db_scores_std, 
                        db_scores_mean + db_scores_std, 
                        color='b', alpha=0.2)
    ax.set_title("Davies Bouldin Score\n(lower=better)")
    ax.set_xlabel("No. of Clusters") 
    
    if annotate_topN_best_scores:
        # # top_k_indices = sorted(range(len(db_scores)), key=lambda i: db_scores[i], reverse=False)[:k]
        top_k_indices = np.array(db_scores_mean).argsort()[:annotN]
        for i in top_k_indices:
            value = db_scores_mean[i]
            ax.text(i+num_clusters[0], value, f'{num_clusters[i]}', color='red')
    
    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'nongt_metrics_plots/dav_bou.png'), dpi=300)
    if show:
        plt.show()
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ## Hopkins
    ax.plot(num_clusters, hpkn_scores_mean)
    ax.fill_between(num_clusters, 
                        hpkn_scores_mean - hpkn_scores_std, 
                        hpkn_scores_mean + hpkn_scores_std, 
                        color='b', alpha=0.2)
    ax.set_title("Hopkins Statistic\n(1=clustered, 0.5=random, 0=uniforrm)")
    ax.set_xlabel("No. of Clusters")
    ax.set_ylim(-0.1,1.1) 

    plt.tight_layout()    
    if save:
        plt.savefig(os.path.join(save_dir, 'nongt_metrics_plots/hopkn.png'), dpi=300)
    if show:
        plt.show()
    plt.close()

    # plt.tight_layout()    
    # if save:
    #     print("Saving non-ground truth-based clustering metrics plot at {}".format(os.path.join(save_dir, 'feats_clustering_metrics.png')))
    #     plt.savefig(os.path.join(save_dir, 'feats_clustering_nongt_metrics.png'), dpi=300)
    # if show:
    #     plt.show()
    # plt.close()

def cluster_feats(X=None, gt_labels=[],
                #   algorithm='k-Means',
                  model=None,
                  num_clusters = [2,3,5,10], num_runs=10, 
                  make_metrics_plots=True, annotate_topN_best_scores=False, annotN=3,
                  make_silhoutte_plots=True, embed_data_in_2d = False,
                  show=False, save=False, save_dir=None):
    
    if len(gt_labels)!=0:
        print("Ground truth labels provided. Computing ground truth-based clustering metrics along with non-ground truth based metrics.") 
    
    labels = []
    gt_metrics = {}
    nongt_metrics = {}

    gt_metrics_mean, gt_metrics_std = {}, {}
    nongt_metrics_mean, nongt_metrics_std = {}, {}
    
    for n_clusters in num_clusters:
        ## Do 100 runs of kmeans for a given value of k? EXPENSIVE!
        # runIDs = np.random.randint(0, num_runs, size=num_runs)
        run_nongt_scores, run_gt_scores = {},{}
        
        best_run_id_, best_run_lbls_ = None, None
        best_run_sil_score_, best_score_ = 0,0
        best_run_metric_ = 'Calinski-Harabasz'  #'Silhouette'
        for run in range(num_runs):
            # if algorithm == 'k-Means':
            #     clusterer = KMeans(n_clusters=n_clusters, random_state=run)
            clusterer = model.set_params(n_clusters=n_clusters, random_state=run)
            run_lbls_=clusterer.fit_predict(X)

            if len(gt_labels)!=0:
                gt_metrics_ = calc_gt_metrics(gt_labels=gt_labels, cluster_labels=run_lbls_)
                run_gt_scores['run={}'.format(run)] = gt_metrics_
            nongt_metrics_ = calc_nongt_metrics(X=X, cluster_labels=run_lbls_)
            run_nongt_scores['run={}'.format(run)] = nongt_metrics_

            if nongt_metrics_[best_run_metric_] > best_score_:
                best_run_id_ = run
                best_score_ = nongt_metrics_[best_run_metric_]
                best_run_sil_score_ = nongt_metrics_['Silhouette']
                best_run_lbls_ = run_lbls_
            
        print("[k={}] Best run is {} (out of {}) with {} score = {}"\
              .format(n_clusters, best_run_id_, num_runs, best_run_metric_, best_score_))
        cluster_labels = best_run_lbls_
        silhouette_avg = best_run_sil_score_
        labels.append(cluster_labels) 
        
        if len(gt_labels)!=0:
            run_gt_scores = pd.DataFrame.from_dict(run_gt_scores).T
            run_gt_scores = run_gt_scores.agg(['mean', 'std'])
        run_nongt_scores = pd.DataFrame.from_dict(run_nongt_scores).T
        run_nongt_scores = run_nongt_scores.agg(['mean', 'std'])
        # print(run_nongt_scores)
        
        # silhouette_avg = run_nongt_scores.T['mean'].T['Silhouette'].round(3)
        # print(silhouette_avg)

        # print(sys.exit())

        if len(gt_labels)!=0:
            gt_metrics_mean['k={}'.format(n_clusters)] = run_gt_scores.T['mean']
            gt_metrics_std['k={}'.format(n_clusters)] = run_gt_scores.T['std']
        nongt_metrics_mean['k={}'.format(n_clusters)] = run_nongt_scores.T['mean']
        nongt_metrics_std['k={}'.format(n_clusters)] = run_nongt_scores.T['std']
        
        # print(nongt_metrics_mean)
        # print(nongt_metrics_std)
        # print(sys.exit())
            
        ## -------------- Single Run -----------------
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        # cluster_labels = clusterer.fit_predict(X)
        # labels.append(cluster_labels)
        # print(labels)
        # if gt_labels:
        #     gt_metrics['k={}'.format(n_clusters)] = calc_gt_metrics(gt_labels=gt_labels, cluster_labels=cluster_labels)
        # nongt_metrics_ = calc_nongt_metrics(X=X, cluster_labels=cluster_labels)
        # nongt_metrics['k={}'.format(n_clusters)] = nongt_metrics_
        # silhouette_avg = nongt_metrics_['Silhouette']
        ## -------------------------------------------
        
        if make_silhoutte_plots:
            # Create a subplot with 1 row and 2 columns
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            # fig.set_size_inches(18, 7)
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            # ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

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
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            if embed_data_in_2d:
                print("Embedding data in 2-D...")
                X_embedded = TSNE(n_components=2).fit(X)
                # X_embedded = Isomap(n_components=2).fit_transform(X)
                # X_embedded = SpectralEmbedding(n_components=2).fit_transform(X)
                # X_embedded = PCA(n_components=2).fit_transform(X)
                ax2.scatter(
                    X_embedded[:, 0], X_embedded[:, 1], marker=".", s=100, lw=0, alpha=1, c=colors, edgecolor="k"
                )
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

            ax2.set_title("The visualization of the clustered data.")
            # ax2.set_xlabel("TSNE axis-1")
            # ax2.set_ylabel("TSNE axis-2")
            # ax2.set_xlabel("Feature space for the 1st feature")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )
            
            plt.tight_layout()
            if save:
                print("Saving silhouette plot for k={} at {}"\
                      .format(n_clusters, os.path.join(save_dir,'silhouette_plots/{}_silhouette_plot.png'.format(n_clusters))))
                plt.savefig(os.path.join(save_dir,'silhouette_plots/{}_silhouette_plot.png'.format(n_clusters)), dpi=300)
            if show:
                plt.show()
            plt.close()

    labels = pd.DataFrame(data=np.array(labels).T, 
                          columns=['k={}'.format(k) for k in num_clusters]).rename_axis('id').reset_index()
    
    if len(gt_labels)!=0:
        # gt_metrics = pd.DataFrame.from_dict(gt_metrics).T
        gt_metrics_mean = pd.DataFrame.from_dict(gt_metrics_mean).T
        gt_metrics_std = pd.DataFrame.from_dict(gt_metrics_std).T
        gt_metrics = {'mean':gt_metrics_mean, 'std':gt_metrics_std}
        
    # nongt_metrics = pd.DataFrame.from_dict(nongt_metrics).T
    nongt_metrics_mean = pd.DataFrame.from_dict(nongt_metrics_mean).T
    nongt_metrics_std = pd.DataFrame.from_dict(nongt_metrics_std).T

    nongt_metrics = {'mean':nongt_metrics_mean, 'std':nongt_metrics_std}
    
    if make_metrics_plots:
        plot_nongt_metrics(num_clusters=num_clusters,
                        nongt_metrics=nongt_metrics, 
                        annotate_topN_best_scores=annotate_topN_best_scores, annotN=annotN,
                        save_dir=save_dir, show=show, save=save)
        if len(gt_labels)!=0:
            plot_gt_metrics(num_clusters=num_clusters,
                            gt_metrics=gt_metrics, 
                            annotate_topN_best_scores=annotate_topN_best_scores, annotN=annotN,
                            save_dir=save_dir, show=show, save=save)

    return labels, nongt_metrics, gt_metrics

def eval_clustering(X=None, gt_labels=[], 
                    num_clusters = [2,3,5,10], num_runs=10, 
                    model=None,
                    # algorithm='k-Means',
                    dataset_type='features',
                    distance_metric='euclidean',
                    show=False, save=False, RESULTS_DIR=None, 
                    make_metrics_plots=True, annotate_topN_best_scores=True, annotN=3,
                    make_silhoutte_plots=False, embed_data_in_2d=False):
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR+'/silhouette_plots')
        os.makedirs(RESULTS_DIR+'/gt_metrics_plots')
        os.makedirs(RESULTS_DIR+'/nongt_metrics_plots')
    
    # if dataset_type == 'features':
    return cluster_feats(X=X, gt_labels=gt_labels, 
                        model=model,
                        # algorithm=algorithm,
                        num_clusters=num_clusters, num_runs=num_runs,
                        show=show, save=save, save_dir=RESULTS_DIR,
                        make_metrics_plots=make_metrics_plots, annotate_topN_best_scores=annotate_topN_best_scores, annotN=annotN,
                        make_silhoutte_plots=make_silhoutte_plots, embed_data_in_2d=embed_data_in_2d)