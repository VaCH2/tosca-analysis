import itertools
import os
import pickle
import pandas as pd
from classes.clusterconfigurator import ClusterConfigurator
from classes.data import Data
from imblearn.over_sampling import RandomOverSampler 


class SmellEvaluator():

    root_folder = os.path.dirname(os.path.dirname( __file__ ))
    results_folder = os.path.join(root_folder, 'results', 'clustering_models')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

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

    def getSmells(self):
        smells = pd.read_excel('results/labeling/to_label.xlsx', sheet_name='Sheet1', usecols='B:H', nrows=685, index_col=0)
        return smells.astype(bool)

    def oversampleData(self, df):
        oversample = RandomOverSampler(sampling_strategy=0.5, random_state=1)
        oversampleDf, _ = oversample.fit_resample(df, df['smell'])
        return oversampleDf

    def constructDf(self, smell):
        smellSeries = self.getSmells()[smell].rename('smell')
        df = self.getData()
        df = df.merge(smellSeries, how='inner', left_index=True, right_index=True)
        df = df.reset_index()
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
        df = df.drop('index', axis=1)
        scoreDict = {}

        for config in configs:
            try:
                configInstance = self.getPickle(self.smell, config)
            except (OSError, IOError):
                configInstance = ClusterConfigurator(df, config)
                self.setPickle(self.smell, config, configInstance)
            
            scores = configInstance.scores
            distribution = configInstance.labels['cluster'].value_counts()
            scores['smellySize'] = distribution.loc[True]
            scores['soundSize'] = distribution.loc[False]
            scoreDict[self.c2s(self.smell, config)] = scores

        scoreDf = pd.DataFrame.from_dict(scoreDict, orient='index', columns=['sc', 'ch', 'db', 'precision', 'mcc', 'ari', 'soundSize', 'smellySize'])
        evalDf = self.scoreAggregation(scoreDf, config)
        evalDf['total_score_percentage'] = (evalDf['total_score'] / 1920) * 100
        return evalDf


    def scoreAggregation(self, scoreDf, config):
        evalDf = scoreDf.copy(deep=True)
        evalDf['total_score'] = 0
        evalDf = evalDf.reset_index()

        for pm in scoreDf.columns:
            if pm is 'db':
                evalDf = evalDf.sort_values(by=pm, ascending=True)
            elif pm in ['smellySize', 'soundSize']:
                break
            else:
                evalDf = evalDf.sort_values(by=pm, ascending=False)
            evalDf = evalDf.reset_index(drop=True)
            evalDf['total_score'] = evalDf['total_score'] + evalDf.shape[0] - evalDf.index.values

        evalDf = evalDf.set_index('index')
        evalDf = evalDf.sort_values(by='total_score', ascending=False)
        return evalDf

    def getTopConfig(self, evalDf):
        return evalDf.iloc[0]