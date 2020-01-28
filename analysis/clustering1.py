import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

from data import Data
from prep import Preprocessing

class Clustering():



    def __init__(self, data, k, threshold, PCA=False):
        '''Provide data class, k clusters, and a threshold for the cluster balance'''
        self.df = data.df
        self.threshold = threshold
        df = data.df

        algos = {'kmeans' : KMeans, \
            'specclu' : SpectralClustering, 
            'aggclu' : AgglomerativeClustering} 
            
        nonk_algos = {'meanshift' : MeanShift, \
            'dbscan' : DBSCAN,
            'optics' : OPTICS} 

        metrics = {'silhouette' : silhouette_score, \
            'davies' : davies_bouldin_score, 
            'calinski' : calinski_harabasz_score}

        if PCA:
            df = self.__perform_PCA(self.df)

        try:
            self.n_clusters = pickle.load(open('../temp_data/tosca_and_general_all_n_clusters', 'rb'))
                
        except (OSError, IOError) as e:
            self.n_clusters = self.__create_total_cluster_dfs(df, algos, k, func='k')
            pickle.dump(self.n_clusters, open('../temp_data/tosca_and_general_all_n_clusters', 'wb'))

        try:
            self.nonk_clusters = pickle.load(open('../temp_data/tosca_and_general_all_nonk_clusters', 'rb'))
                
        except (OSError, IOError) as e:
            self.nonk_clusters = self.__create_total_cluster_dfs(df, nonk_algos, k, func='nonk')
            pickle.dump(self.nonk_clusters, open('../temp_data/tosca_and_general_all_nonk_clusters', 'wb'))

    def __scale_df(self, df):
        copy_df = df.copy()
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(copy_df)
        copy_df.loc[:,:] = scaled_values
        return copy_df

    def __perform_PCA(self, df):
        pca = PCA(n_components=5)
        X_std = self.__scale_df(self.df.copy())
        X_std = X_std.to_numpy()
        principalComponents = pca.fit_transform(X_std)
        PCA_components = pd.DataFrame(principalComponents)
        return PCA_components


    def __clustering(self, df, algo, k):
        scaled_df = self.__scale_df(df)
        model = algo(n_clusters=k)
        return model.fit_predict(scaled_df)


    def __nonk_clustering(self, df, algo):
        scaled_df = self.__scale_df(df)
        model = algo()
        return model.fit_predict(scaled_df)
    

    def __cluster_balance(self, cluster_column, threshold):
        '''Calculates the proportion of each cluster as a percentage of the total size.
        It returns True if each cluster passes the size threshold. Otherwise False.'''
        
        count = cluster_column.value_counts()
        percentages = [count.iloc[ix] / count.sum() > threshold for ix in range(len(count))]
        
        return all(elem == True for elem in percentages)


    def __create_total_cluster_dfs(self, df, algos, k, func):

        copy_df = df.copy()
        new_dfs = {''.join([algo_name]) : copy_df for algo_name in algos.keys()}
        cluster_dfs = {}
        for algo_name, df in new_dfs.items():
            algo_df = df.copy()
            if func == 'k':
                algo_df['cluster'] = self.__clustering(algo_df, algos[algo_name], k)
            elif func == 'nonk':
                algo_df['cluster'] = self.__nonk_clustering(algo_df, algos[algo_name])
            if self.__cluster_balance(algo_df['cluster'], self.threshold):
                cluster_dfs[algo_name] = algo_df

        return cluster_dfs


data = Data('tosca_and_general', 'all')
data = Preprocessing(data, chi=True)
inst = Clustering(data, 2, 0.05, PCA=False)

#%%
import pickle
pickle.dump(data.df, open('../temp_data/{}_{}_braycurtisdistance'.format('tosca_and_general', 'all'), 'wb'))

#%%

try:
    self.df = pickle.load(open('../temp_data/{}_{}_braycurtisdistance'.format(metrics_type, dataset), 'rb'))
    
except (OSError, IOError) as e:
    self.df = self.__transform_distance()
    pickle.dump(self.df, open('../temp_data/{}_{}_braycurtisdistance'.format(metrics_type, dataset), 'wb'))