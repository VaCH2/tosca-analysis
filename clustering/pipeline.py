import pandas as pd
from clustering.datatrans import Dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans 
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
#from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
#from sklearn.metrics import homogeneity_score #Deze kan niet omdat je hier een ground truth voor nodig hebt

df = Dataset(10, True).getDf
df = normalize(df)

k_sizes = [2,3,4,5,6,7,8]

algos = {'kmeans' : KMeans, \
    'specclu' : SpectralClustering, 
    'aggclu' : AgglomerativeClustering} #(), MeanShift(), SpectralClustering(), AgglomerativeClustering(), DBSCAN(), OPTICS()]

metrics = {'silhouette' : silhouette_score, \
    'davies' : davies_bouldin_score, 
    'calinski' : calinski_harabasz_score}

def clustering(df, algo, k):
    model = algo(n_clusters=k)
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


