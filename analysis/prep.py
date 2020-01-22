import pandas as pd
import numpy as np
from data import Data
import pickle

class Preprocessing():
    def __init__(self, data, constants=None, corr=None, pca=False, anomalies=False):
        '''For constants and corr please provide the list of to be dropped columns'''
        self.df = data.df
        self.anomalypercentage = None
        to_drop = []
        if corr != None:
            to_drop.extend(corr)

        if constants != None:
            to_drop.extend(constants)
        
        to_drop = list(set(to_drop))
        # print('len dropped columns: ', len(to_drop))
        # print('dropped columns: ', to_drop)
        
        self.df = self.df.drop(to_drop, axis=1)
        
        if anomalies != None:
            self.anomalypercentage = self.filter_anomalies()


    
    def filter_anomalies(self):
        pass



#df = pickle.load(open('../temp_data/{}_{}_raw_df'.format('tosca_and_general', 'all'), 'rb'))

# df = Data('tosca_and_general', 'all').df
# test = Selection(df)

# for dat in ['all', 'industry', 'example', 'a4c', 'forge', 'puccini']:
#     for met in ['general', 'tosca', 'tosca_and_general']:
#         Data(met, dat) 