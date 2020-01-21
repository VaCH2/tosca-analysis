import pandas as pd
import numpy as np
from data import Data
import pickle

class Selection():
    def __init__(self, df, filetype=None, anomalies=None):
        self.df = df
        self.typefilterpercentage = None
        self.anomalypercentage = None
        if filetype != None:
            self.typefilterpercentage = self.filter_filetype(filetype)
        
        if anomalies != None:
            self.anomalypercentage = self.filter_anomalies()


    def filter_filetype(self, filetype):
        '''Filter on the file type. A file could be a service template, or containing
        custom type definitions, both or none of these two. It assigns the filtered df
        to self.df and assings the filtered percentage to typefilterpercentage'''

        if not filetype in ['topology', 'custom', 'both', 'none']:
            raise ValueError('Enter a valid file type (topology, custom, both, none)')

        cus_df = self.df[['cdnt_count', 'cdrt_count', 'cdat_count', 
        'cdct_count', 'cddt_count', 'cdgt_count', 'cdit_count', 'cdpt_count']]

        non_df = cus_df[(cus_df == 0).all(1)] 
        self.df['custom_def'] = [False if x in non_df.index else True for x in self.df.index]

        if filetype == 'topology':
            result = self.df[(self.df['ttb_check'] == 1) & (self.df['custom_def'] == False)]

        if filetype == 'custom':
            result = self.df[(self.df['ttb_check'] == 0) & (self.df['custom_def'] == True)]

        if filetype == 'both':
            result = self.df[(self.df['ttb_check'] == 1) & (self.df['custom_def'] == True)]

        if filetype == 'none':
            result = self.df[(self.df['ttb_check'] == 0) & (self.df['custom_def'] == False)]

        result = result.drop('custom_def', axis=1)
        self.df = result
        return len(result)/len(cus_df)

    
    def filter_anomalies(self):
        pass



#df = pickle.load(open('../temp_data/{}_{}_raw_df'.format('tosca_and_general', 'all'), 'rb'))

df = Data('tosca_and_general', 'all').df
test = Selection(df)

# for dat in ['all', 'industry', 'example', 'a4c', 'forge', 'puccini']:
#     for met in ['general', 'tosca', 'tosca_and_general']:
#         Data(met, dat) 