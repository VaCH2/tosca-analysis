from stats import Stats
from anomaly import AnomalyDetector
from utils import scale_df

import numpy as np
import pandas as pd
import random

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.cluster import AgglomerativeClustering
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import adjusted_rand_score


class ClusterConfigurator():
    def __init__(self, df, config):
        self.df = df
        self.name = str(config)
        self.ex_spr = config[0]
        self.ex_cor = config[1]
        self.ex_out = config[2]
        self.pca = config[3]
        self.distance = config[4]
        self.algorithm = config[5]
        self.model, self.labels = self.modelConfiguration(self.df)
        self.scores = self.performanceEvaluation(self.labels)
        self.cardinality = self.getCardinality(self.labels)


    def handleConstants(self, df):
        df = df.copy(deep=True)
        return df.loc[:, df.nunique() != 1]


    def handleSparsity(self, df):
        sparsity = Stats(df).featuresparsity
        sparse_measurements = sparsity[sparsity['sparsity'] > 0.9].index.values
        return df.drop(sparse_measurements, axis=1)


    def handleCorrelations(self, df):
        matrix = df.corr()
        matrix = matrix.where(~np.tril(np.ones(matrix.shape)).astype(np.bool))
        matrix = matrix.stack()
        matrix = matrix[matrix > 0.8]
        matrix = matrix.to_frame()
        first_level = list(matrix.index.get_level_values(0))
        return df.drop(list(set(first_level)), axis=1)


    def handleAnomalies(self, df):
        copy_df = df.copy(deep=True)
        outliers = AnomalyDetector(copy_df).outliers.index.values
        return df.drop(outliers, axis=0)

    def handlePca(self, df):
        pca = PCA(n_components=15)
        X_std = scale_df(df)
        principalComponents = pca.fit_transform(X_std)
        return pd.DataFrame(principalComponents, index=df.index.values)

    def handleDistanceFunction(self, dist, df):
        X_std = scale_df(df)
        distances = pairwise_distances(X_std, metric=dist)
        return pd.DataFrame(data=distances, index=df.index, columns=df.index)

    def handleAlgorithm(self, algorithm, distancematrix, df):
        df = df.copy(deep=True)
        if algorithm[0] == 'kmedoids':
            medoidInitiations = random.sample(range(0, df.shape[0]), 2)
            model = kmedoids(distancematrix.to_numpy(), medoidInitiations, data_type='distance_matrix')
            model.process()
            results = model.get_clusters()
            # kmedoids model is not pickable, so as a workaround discarded here
            model = None
            labels = pd.Series(index=df.index)
            labels.iloc[results[0]] = 0
            labels.iloc[results[1]] = 1

        if algorithm[0] == 'specclu':
            if distancematrix is None:
                model = SpectralClustering(n_clusters=2)
                labels = model.fit_predict(df.to_numpy())
            else:
                model = SpectralClustering(n_clusters=2, affinity='precomputed')
                labels = model.fit_predict(distancematrix.to_numpy())

        if algorithm[0] == 'agglo':
            model = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage=algorithm[1])
            labels = model.fit_predict(distancematrix.to_numpy())

        if algorithm[0] == 'gm':
            model = GaussianMixture(n_components=2, covariance_type=algorithm[1])
            labels = model.fit_predict(df.to_numpy())
        
        df['cluster'] = labels
        return model, df['cluster']

    def labelMatching(self, smellLabels, predLabels):
        labelDf = pd.concat([smellLabels, predLabels], axis=1)
        countDf = pd.DataFrame(index=[True, False])
        for predLabel in predLabels.unique():
            count = labelDf[labelDf['cluster'] == predLabel]['smell'].value_counts()
            countDf[predLabel] = count

        if countDf.loc[True, 1] + countDf.loc[False, 0] > countDf.loc[True, 0] + countDf.loc[False, 1]:
            labelDf['cluster'] = labelDf['cluster'].map({1 : True, 0 : False})
        else:
            labelDf['cluster'] = labelDf['cluster'].map({1 : False, 0 : True})
        
        return labelDf

    def modelConfiguration(self, df):
        df = df.copy(deep=True)
        smellLabels = df['smell']
        df = df.drop('smell', axis=1)   

        df = self.handleConstants(df)
        if self.ex_spr:
            df = self.handleSparsity(df)
        if self.ex_cor:
            df =self.handleCorrelations(df)
        if self.ex_out:
            df = self.handleAnomalies(df)
        if self.pca:
            df = self.handlePca(df)
        
        if self.distance is None:
            distancematrix = None
        else:
            distancematrix = self.handleDistanceFunction(self.distance, df)

        model, predLabels = self.handleAlgorithm(self.algorithm, distancematrix, df)

        labels = self.labelMatching(smellLabels, predLabels)
        df = df.merge(labels, how='left', left_index=True, right_index=True)
        return model, df



    def performanceEvaluation(self, df):
        df = df.copy(deep=True)

        y_true = df['smell']
        y_true = y_true.map({True: 1, False: 0})

        y_pred = df['cluster']
        y_pred = y_pred.map({True: 1, False: 0})

        df = df.drop(['smell', 'cluster'], axis=1)

        internalPerformanceMeasurements = {
            'sc' : silhouette_score,
            'ch' : calinski_harabasz_score,
            'db' : davies_bouldin_score
        }

        externalPerformanceMeasurements = {
            'precision': precision_score,
            'mcc': matthews_corrcoef,
            'ari': adjusted_rand_score
        }

        performanceScores = {}

        for name, module in internalPerformanceMeasurements.items():
            try:
                performanceScores[name] = module(X=df.to_numpy(), labels=y_pred)
            except:
                performanceScores[name] = np.nan

        for name, module in externalPerformanceMeasurements.items():
            try:
                performanceScores[name] = module(y_true, y_pred)
            except:
                performanceScores[name] = np.nan

        return performanceScores

    def getCardinality(self, df):
        valueCounts = df['cluster'].value_counts()
        return valueCounts.max() / df.shape[0]

    def getEuclideanDistance(self, clusterOne, clusterTwo):
        distances = euclidean_distances(clusterOne, clusterTwo)
        return distances.sum()

    def getClusterToClusterMatching(self, initModelDf, sampleModelDf):

        distances = []
        for i in initModelDf['cluster'].unique():
            for j in sampleModelDf['cluster'].unique():
                xInit = initModelDf[initModelDf['cluster'] == i]
                xInit = xInit.drop('cluster', axis=1).to_numpy()

                xSample = sampleModelDf[sampleModelDf['cluster'] == j]
                xSample = xSample.drop('cluster', axis=1).to_numpy()

                distance = self.getEuclideanDistance(xInit, xSample)
                distances.append([(i, j), distance])

        minDistance = min(distances, key=lambda x: x[1])
        if minDistance[0] == (True, 0) or minDistance[0] == (False, 1):
            sampleModelDf['cluster'] = sampleModelDf['cluster'].map({0 : True, 1 : False})
        else:
            sampleModelDf['cluster'] = sampleModelDf['cluster'].map({0 : False, 1 : True})

        return sampleModelDf

    
    def getStability(self):
        '''Determine configuration stability. We perform a 20 times resampling. When this results in an 
        mean Adjusted Rand Index of lower than 0.9, we initialize again. Max 3 initializations.
        Returns the best found mean Adjusted Rand Index together the list of actual ARIs'''

        it = 0
        meanAri = 0
        bestIteration = []

        while meanAri < 0.9 and it < 3:
            if it == 0:
                initModelDf = self.labels
                initModelDf = initModelDf.drop('smell', axis=1)
            else:
                _, initModelDf = self.modelConfiguration(self.df)
                initModelDf = initModelDf.drop('smell', axis=1)
            
            scoresAri = []

            for i in range(10):
                if i % 5 == 0:
                    print('Iteration: ', it, '---- Epoch: ', i)

                sample = initModelDf.drop('cluster', axis=1).sample(frac=0.7)

                if self.distance is None:
                    distancematrix = None
                else:
                    distancematrix = self.handleDistanceFunction(self.distance, sample)

                _, predLabels = self.handleAlgorithm(self.algorithm, distancematrix, sample)
                sampleModelDf = sample.merge(predLabels, how='left', left_index=True, right_index=True)
                sampleModelDf = self.getClusterToClusterMatching(initModelDf, sampleModelDf)

                ari = adjusted_rand_score(initModelDf['cluster'].loc[sampleModelDf.index], sampleModelDf['cluster'])
                scoresAri.append(ari)

            meanAri = np.mean(scoresAri)
            bestIteration.append([meanAri, scoresAri])
            it += 1

        return max(bestIteration, key=lambda x: x[0])