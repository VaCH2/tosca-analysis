import pandas as pd
import numpy as np
from classes.data import Data
from classes.utils import scale_df
import pickle
from scipy.stats import chi2_contingency as chi2
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from classes.anomaly import AnomalyDetector

class Preprocessing():
    def __init__(self, data, constants=None, corr=None, pca=False, anomalies=None, customdistance=None):
        '''For constants and corr please provide the list of to be dropped columns. \
        For anomalies provide the cutoff point. For custom distance provide valid distance \
            (spearman, braycurtis, cosine, l1)'''

        if isinstance(data, Data):
            self.df = data.df
        elif isinstance(data, pd.DataFrame):
            self.df = data

        to_drop = []
        if corr != None:
            to_drop.extend(corr)

        if constants != None:
            to_drop.extend(constants)
        
        to_drop = list(set(to_drop))
        
        self.df = self.df.drop(to_drop, axis=1)
        
        if anomalies != None:
            self.df = self.filter_anomalies(self.df, anomalies)

        if customdistance in ['spearman', 'braycurtis', 'cosine', 'l1']:
            try:
                self.df = pickle.load(open('../temp_data/tosca_and_general_all_{}_anomalies_{}'.format(customdistance, anomalies), 'rb'))
                
            except (OSError, IOError) as e:
                self.df = self.__transform_distance(customdistance)
                pickle.dump(self.df, open('../temp_data/tosca_and_general_all_{}_anomalies_{}'.format(customdistance, anomalies), 'wb'))
        else:
            print('Invalid distance function! Valid funtions are: spearman, braycurtis, cosine, l1')


    def __transform_distance(self, customdistance):
        df = self.df.copy()
        scaled = scale_df(df)

        if customdistance == 'spearman':
            matrix = pd.DataFrame(index=self.df.index, columns=self.df.index)
            combs = [ ((x,y), spearmanr(scaled.loc[[x], :].to_numpy().flatten(),scaled.loc[[y], :].to_numpy().flatten())) for x in list(self.df.index) for y in list(self.df.index)]
            for element in combs:
                matrix.loc[element[0][0], element[0][1]] = element[1][0]

        else:
            distances = pairwise_distances(scaled, metric=customdistance)
            matrix = pd.DataFrame(data=distances, index=self.df.index, columns=self.df.index)

        # matrix = pd.DataFrame(index=self.df.index, columns=self.df.index)
        # # Werkt niet voor chi2 omdat de data veel te sparse is.. Stond ergens een minimale count van 5 per kolom. 
        # #Nu maar braycurtis genomen maar weet niet of hier wat haken en ogen aan zitten..
        # # Hier staat wat uitgelegd over de 2: http://84.89.132.1/~michael/stanford/maeb5.pdf 
        # combs = [ ((x,y), braycurtis(self.df.loc[[x], :].to_numpy(),self.df.loc[[y], :].to_numpy())) for x in list(self.df.index) for y in list(self.df.index)]
        # for element in combs:
        #     matrix.loc[element[0][0], element[0][1]] = element[1]
        return matrix




    def filter_anomalies(self, df, cutoff):
        ix = AnomalyDetector(df, cutoff).outliers
        print('Number of anomalies: ', len(ix))

        to_drop = [i for i in ix.index if i in df.index]
        df = df.drop(to_drop)

        print('After dropping {} datapoints, there are {} datapoints left'.format(len(to_drop), df.shape[0]))
        
        # filtered_datasets = []
        # for dataset in datasets:
            
        #     print('originele grootte df: ', dataset.df.shape[0])
        #     print('aantal anomalies: ', len(to_drop))
        #     new_df = dataset.df.drop(to_drop)
        #     print('nieuwe lengte: ', new_df.shape[0])

        #     filtered_datasets.append(Stats(new_df))

        return df