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








#Dit moet de daadwerkelijke hoofdclass worden waarin de gehele pipeline wordt uigevoerd.
#class SmellEvalutor()

#Hier dan nog de balancing bij, cardinality en stability bij  

#!! De DF die je naar de clusterconfigurator stuurt moet een column 'smell' hebben met het originele True/False label
#Deze moet uit de smellsDf komen!

import itertools
import os
import pickle
import pandas as pd
from clusterconfigurator import ClusterConfigurator
from data import Data
from imblearn.over_sampling import RandomOverSampler 


class SmellEvaluator():

    root_folder = os.path.dirname(os.path.dirname( __file__ ))
    results_folder = os.path.join(root_folder, 'results', 'clustering_models')

    def __init__(self, smell):        
        self.smell = smell
        self.df = self.constructDf(self.smell)
        self.configs = self.getConfigurations()
        self.evalDf = self.configCalculationAndEvaluation(self.df, self.configs)
        self.topconfig = self.getTopConfig(self.evalDf)


    def getData(self):
        data = Data().dfs.get('all')
        data = data.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
        return data

    #Dit moet de gelabelde set zijn! Wellicht als csv inladen
    def getSmells(self):
        smells = pickle.load(open(os.path.join(self.root_folder, 'temp_data', 'smells_df'), 'rb'))
        smells = smells.drop(r'SeaCloudsEU\tosca-parser\Industry\normative_types.yaml')
        return smells.astype(bool)

    def oversampleData(self, df):
        oversample = RandomOverSampler(sampling_strategy='minority', random_state=1)
        oversampleDf, _ = oversample.fit_resample(df, df['smell'])
        return oversampleDf

    #.head weghalen!
    def constructDf(self, smell):
        smellSeries = self.getSmells()[smell].rename('smell').head(100)
        df = self.getData()
        df = df.merge(smellSeries, how='right', left_index=True, right_index=True)
        df = self.oversampleData(df)
        return df

    def getConfigurations(self):
        ex_out = ex_cor = pca = ex_spr = [True, False]
        prep = [ex_out, ex_cor, pca, ex_spr]
        algos = [
            ('agglo', 'complete'), ('agglo', 'average'), ('agglo', 'single'), 
            ('kmedoids', None), ('specclu', None), 
            ('gm', 'full'), ('gm', 'tied'), ('gm', 'diag'), ('gm', 'spherical')
            ]
        distance = ['braycurtis', 'cosine', 'l1', None]

        configurations = []
        for algo in algos:
            if algo[0] is 'gm':
                distance = [None]
            elif algo[0] is 'specclu':
                distance = ['braycurtis', 'cosine', 'l1', None]
            else:
                distance = ['braycurtis', 'cosine', 'l1']

            distance_perm = list(itertools.product(*[distance, [algo]]))
            prep_perm = list(itertools.product(*prep))
            total_perm = list(itertools.product(*[prep_perm, distance_perm]))
            total_perm = [(t[0][0], t[0][1], t[0][2], t[0][3], t[1][0], t[1][1]) for t in total_perm]
            
            configurations.extend(total_perm)
        
        return configurations

    def c2s(self, smell, config):
        '''Config to string'''
        return f'smell={smell}_exout={config[0]}_excor={config[1]}_pca={config[2]}_exspr={config[3]}_{config[4]}_{config[5][0]}_{config[5][1]}'

    def getPickle(self, smell, config):
        return pickle.load(open(os.path.join(self.root_folder, 'temp_data', 'clustering', self.c2s(smell, config)), 'rb'))

    def setPickle(self, smell, config, instance):
        pickle.dump(instance, open(os.path.join(self.root_folder, 'temp_data', 'clustering', self.c2s(smell, config)), 'wb'))


    def configCalculationAndEvaluation(self, df, configs):
        scoreDict = {}

        for config in configs[:5]:
            try:
                configInstance = self.getPickle(self.smell, config)
            except (OSError, IOError):
                configInstance = ClusterConfigurator(df, config)
                self.setPickle(self.smell, config, configInstance)
            
            scoreDict[config] = configInstance.scores

        scoreDf = pd.DataFrame.from_dict(scoreDict, orient='index', columns=['sc', 'ch', 'db', 'precision', 'mcc', 'ari'])
        scoreDf = scoreDf.set_index(pd.Index(scoreDict.keys()))
        evalDf = self.scoreAggregation(scoreDf, config)  
        return evalDf


    def scoreAggregation(self, scoreDf, config):
        evalDf = scoreDf.copy(deep=True)
        evalDf['total_score'] = 0

        for pm in scoreDf.columns:
            if pm is 'db':
                evalDf = evalDf.sort_values(by=pm, ascending=True)
            else:
                evalDf = evalDf.sort_values(by=pm, ascending=False)
            evalDf = evalDf.reset_index(drop=True)
            evalDf['total_score'] = evalDf['total_score'] + evalDf.shape[0] - evalDf.index.values

        evalDf = evalDf.set_index(scoreDf.index)
        return evalDf

    def getTopConfig(self, evalDf):
        return evalDf.iloc[0]


#Loop [:5] nog weghalen!
test = SmellEvaluator('db')

#Op deze manier kan je dan door een 
topModel = SmellEvaluator('db').getPickle('db', test.evalDf.iloc[4].name)
topModel.labels

#Als we dan de stability willen bereken moeten we m ff opnieuw aanroepen(nu tenminste, kan evt wel in loop)
#top_db = ClusterConfigurator()
