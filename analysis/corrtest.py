from data import Data
import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from utils import scale_df

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyclustering.cluster.kmedoids import kmedoids

df = Data().dfs.get('all')

# for col in df.columns:
#     s = df[col]
#     s = s[s > 0]
#     stat, p = shapiro(s.values)
#     print('measurement: ', col)
#     print('stat: ', stat, 'p-value: ', p)


# kmean = KMeans(n_clusters=2)
# kmean_results = kmean.fit_predict(df.values)
# kmean_sil = silhouette_score(df.values, kmean_results)


# X_std = scale_df(df)
# dm = pairwise_distances(X_std, metric='cosine')

# kmed = kmedoids(dm, [4,865], data_type='distance_matrix')
# kmed.process()
# kmed_results = kmed.get_clusters()
# s = pd.Series(index=df.index)
# s.iloc[kmed_results[0]] = 0
# s.iloc[kmed_results[1]] = 1

# df = df[df['loc_count'] < 20000]

# for col in df.columns:
#     x = df[['loc_count', col]]
#     x = x[x[col] > 0]
#     plot = x.plot.scatter(x='loc_count', y=col)





