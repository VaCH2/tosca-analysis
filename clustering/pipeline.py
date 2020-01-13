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
    scores = pd.DataFrame(index=index, columns=range(20))

    for algo_name, algo in nonk_algos.items():
        result = nonk_clustering(df, algo)
        max_cluster = max(np.unique(result))
        for metric_name, metric in metrics.items():
            if max_cluster > 0:
                score = metric(df, result)
                scores.loc[(algo_name, metric_name), max_cluster] = score
    return scores

def create_cluster_dfs(original_df, algos, k, func=clustering):
    new_dfs = {''.join([algo_name]) : original_df for algo_name in algos.keys()}
    cluster_dfs = {}
    for algo_name, algo_df in new_dfs.items():
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



original_df = Dataset(0, 'all').getDf
df = original_df #normalize(original_df)

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

# %%
k_scores = calculate_cluster_score(df, k_sizes, algos, metrics)
nonk_scores = calculate_nonk_cluster_score(df, nonk_algos, metrics)

#%% Based on the cluster score, it turns out k=2 is the optimal cluster size
clusters = create_cluster_dfs(original_df, algos, 2)
results = get_cluster_stats(clusters)
#results.to_excel('cluster_stats.xlsx')

#%% Based on the nonk cluster score, it turns out k=17 the max cluster size
clusters = create_cluster_dfs(original_df, nonk_algos, 17, func=nonk_clustering)
results = get_cluster_stats(clusters)
results.to_excel('nonk_cluster_stats.xlsx')

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def create_total_cluster_dfs(original_df, algos, k, func=clustering):
    new_dfs = {''.join([algo_name]) : original_df for algo_name in algos.keys()}
    cluster_dfs = {}
    for algo_name, algo_df in new_dfs.items():
        if func == clustering:
            algo_df['cluster'] = func(algo_df, algos[algo_name], k)
        elif func == nonk_clustering:
            algo_df['cluster'] = func(algo_df, algos[algo_name])
        cluster_dfs[algo_name] = algo_df
    return cluster_dfs

def feature_extraction(df, predictor):
    y = df['cluster']
    X = df.drop(columns=['cluster'])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    predictor.fit(X=X_train, y=y_train)
    score = predictor.score(X_test, y_test)
    imps = predictor.feature_importances_
    feature_importances = sorted(zip(X_test, imps), reverse=True, key=lambda x: x[1])
    feature_importances.append(('pred_score', score))        
    return feature_importances

#%%
all_clusters = create_total_cluster_dfs(original_df, algos, 2)
all_nonk_clusters = create_total_cluster_dfs(original_df, nonk_algos, 7, func=nonk_clustering)
all_clusters.update(all_nonk_clusters)
    
#%%

feature_importances = pd.DataFrame()

for clu_algo, df in all_clusters.items():
    importance = feature_extraction(df, RandomForestClassifier())
    feature_importances[clu_algo] = importance

feature_importances