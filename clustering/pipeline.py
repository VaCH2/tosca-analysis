#%%

import pandas as pd
import numpy as np
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

original_df = Dataset(7, False).getDf
df = normalize(original_df)

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

def clustering(df, algo, k):
    model = algo(n_clusters=k)
    return model.fit_predict(df)

def nonk_clustering(df, algo):
    model = algo()
    return model.fit_predict(df)

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
    scores = pd.DataFrame(index=index, columns=range(10))

    for algo_name, algo in nonk_algos.items():
        result = nonk_clustering(df, algo)
        max_cluster = max(np.unique(result))
        for metric_name, metric in metrics.items():
            score = metric(df, result)
            scores.loc[(algo_name, metric_name), max_cluster] = score
    return scores

def create_cluster_dfs(original_df, algos, k):
    new_dfs = {''.join([algo_name]) : original_df for algo_name in algos.keys()}
    cluster_dfs = {}
    for algo_name, algo_df in new_dfs.items():
        algo_df['cluster'] = clustering(algo_df, algos[algo_name], k)
        for k_th in range(k):
            cluster_dfs[algo_name + '_cluster_{}'.format(k_th)] = algo_df.loc[algo_df['cluster'] == k_th]
    return cluster_dfs

def get_cluster_stats(df_dict):
    result = pd.DataFrame()
    for name, df in df_dict.items():
        result[name + '_count'] = df.apply(lambda x: np.count_nonzero(x))
        result[name + '_mean'] = df.apply(lambda x: np.mean(x))
        result[name + '_std'] = df.apply(lambda x: np.std(x))
    return result


#%% Based on the cluster score, it turns out k=2 is the optimal cluster size
clusters = create_cluster_dfs(original_df, algos, 2)
results = get_cluster_stats(clusters)
results.to_excel('cluster_stats.xlsx')

#%%
original_df

# %%
nonk_clustering(df, DBSCAN)

# %%
