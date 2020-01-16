#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datatrans import Dataset
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans 
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%config InlineBackend.figure_format='retina'

def scale_df(df):
    copy_df = df.copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(copy_df)
    copy_df.loc[:,:] = scaled_values
    return copy_df

def clustering(df, algo, k):
    scaled_df = scale_df(df)
    model = algo(n_clusters=k)
    return model.fit_predict(scaled_df)

def nonk_clustering(df, algo):
    scaled_df = scale_df(df)
    model = algo()
    return model.fit_predict(scaled_df)

def calculate_cluster_score(df, k_sizes, algos, metrics):
    index = pd.MultiIndex.from_product(iterables=[algos.keys(), metrics.keys()], names=['algo', 'evaluation'])
    scores = pd.DataFrame(index=index, columns=k_sizes)

    for k in k_sizes:
        for algo_name, algo in algos.items():
            result = clustering(df, algo, k)
            for metric_name, metric in metrics.items():
                score = metric(df, result)
                scores.loc[(algo_name, metric_name), k] = score
    return scores

def calculate_nonk_cluster_score(df, nonk_algos, metrics):
    index = pd.MultiIndex.from_product(iterables=[nonk_algos.keys(), metrics.keys()], names=['algo', 'evaluation'])
    scores = pd.DataFrame(index=index, columns=range(20))

    for algo_name, algo in nonk_algos.items():
        result = nonk_clustering(df, algo)
        max_cluster = max(np.unique(result))
        for metric_name, metric in metrics.items():
            if max_cluster > 0:
                score = metric(df, result)
                scores.loc[(algo_name, metric_name), max_cluster] = score
    return scores

def create_cluster_dfs(df, algos, k, func=clustering):
    copy_df = df.copy()
    new_dfs = {''.join([algo_name]) : copy_df for algo_name in algos.keys()}
    cluster_dfs = {}
    for algo_name, df in new_dfs.items():
        algo_df = df.copy()
        if func == clustering:
            algo_df['cluster'] = func(algo_df, algos[algo_name], k)
        elif func == nonk_clustering:
            algo_df['cluster'] = func(algo_df, algos[algo_name])
        for k_th in range(-1,k):
            cluster_dfs[algo_name + '_cluster_{}'.format(k_th)] = algo_df.loc[algo_df['cluster'] == k_th]
    return cluster_dfs

def get_cluster_stats(df_dict):
    result = pd.DataFrame()
    for name, df in df_dict.items():
        result[name + '_count'] = df.apply(lambda x: np.count_nonzero(x))
        result[name + '_mean'] = df.apply(lambda x: np.mean(x))
        result[name + '_std'] = df.apply(lambda x: np.std(x))
    return result


def create_total_cluster_dfs(df, algos, k, func=clustering):
    copy_df = df.copy()
    new_dfs = {''.join([algo_name]) : copy_df for algo_name in algos.keys()}
    cluster_dfs = {}
    for algo_name, df in new_dfs.items():
        algo_df = df.copy()
        if func == clustering:
            algo_df['cluster'] = func(algo_df, algos[algo_name], k)
        elif func == nonk_clustering:
            algo_df['cluster'] = func(algo_df, algos[algo_name])
        cluster_dfs[algo_name] = algo_df
    return cluster_dfs

def feature_extraction(df, predictor):
    copy_df = df.copy()
    y = copy_df['cluster']
    X = copy_df.drop(columns=['cluster'])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    predictor.fit(X=X_train, y=y_train)
    score = predictor.score(X_test, y_test)
    imps = predictor.feature_importances_
    feature_importances = sorted(zip(X_test, imps), reverse=True, key=lambda x: x[1])
    feature_importances.append(('pred_score', score))        
    return feature_importances

#%%#########################################################################
##         Load the dataset, scale and keep dataframe structure           ##
############################################################################

original_df = Dataset(1, 'all').getDf

k_sizes = [2,3,4,5,6,7,8]

algos = {'kmeans' : KMeans, \
    'specclu' : SpectralClustering, 
    'aggclu' : AgglomerativeClustering} 
    
nonk_algos = {'meanshift' : MeanShift, \
    'dbscan' : DBSCAN,
    'optics' : OPTICS} 

metrics = {'silhouette' : silhouette_score, \
    'davies' : davies_bouldin_score, 
    'calinski' : calinski_harabasz_score}

#%%#########################################################################
##       evaluate cluster performance for 3 performance measures          ##
############################################################################

k_scores = calculate_cluster_score(original_df, k_sizes, algos, metrics)
nonk_scores = calculate_nonk_cluster_score(original_df, nonk_algos, metrics)
nonk_scores = nonk_scores[nonk_scores.columns[~nonk_scores.isnull().all()]]

#%%#########################################################################
##      Obtain count, mean, std for each individual cluster in each       ##
##      algo, based on the found ideal number for k if possible           ##
############################################################################

sep_k_clusters = create_cluster_dfs(original_df, algos, 2)
sep_k_stats = get_cluster_stats(sep_k_clusters)
sep_k_stats.to_excel('sep_k_cluster_stats.xlsx')

#%%
sep_nonk_clusters = create_cluster_dfs(original_df, nonk_algos, 35, func=nonk_clustering)
sep_nonk_stats = get_cluster_stats(sep_nonk_clusters)
sep_nonk_stats.to_excel('sep_nonk_cluster_stats.xlsx')

#%%#########################################################################
##       Perform all cluster algos on dataset and store in dict           ##
############################################################################

all_clusters = create_total_cluster_dfs(original_df, algos, 2)
all_nonk_clusters = create_total_cluster_dfs(original_df, nonk_algos, 7, func=nonk_clustering)
all_clusters.update(all_nonk_clusters)
    
#%%#########################################################################
##                 Perform feature importance analysis                    ##
############################################################################
feature_importances = pd.DataFrame()

for clu_algo, df in all_clusters.items():
    copy_df = df.copy()
    importance = feature_extraction(copy_df, RandomForestClassifier())
    feature_importances[clu_algo] = importance

feature_importances.to_excel('feature_importance.xlsx')

#%%#########################################################################
##                 PCA analysis and cluster visualisation                 ##
############################################################################
#from  https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
# and  https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60 

# Create a PCA instance: pca
pca = PCA(n_components=20)
X_std = scale_df(original_df.copy())
X_std.values
principalComponents = pca.fit_transform(X_std)

# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

#%%
# Check visually for the existance of clusters
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# %% Visualize per clustering algorithm
for clu_algo, df in all_clusters.items():
    df = df.reset_index(drop=False)
    finalDf = pd.concat([PCA_components, df[['cluster']]], axis = 1)

    #Plot the 2 component PCA with the found clusters by the algorithm
    fig = plt.figure(figsize = (4,4))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('{}'.format(clu_algo), fontsize = 20)

    clusters = [-1, 0, 1, 2, 3, 4, 5]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for cluster, color in zip(clusters,colors):
        indicesToKeep = finalDf['cluster'] == cluster
        ax.scatter(finalDf.loc[indicesToKeep, 0]
                , finalDf.loc[indicesToKeep, 1]
                , c = color
                , s = 50)
    ax.legend(clusters)
    ax.grid()

# %%
