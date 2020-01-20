import pandas as pd
import numpy as np
from data import Data
import pickle

class Selection():
    def __init__(self, df, criteria):
        self.df = df



    def filter_filetype(self, filetype):
        pass

    
    def filter_anomalies(self):
        pass

    def ietsmetresultaten(self):
        pass



#df = pickle.load(open('../temp_data/{}_{}_raw_df'.format('tosca_and_general', 'all'), 'rb'))

df = Data('tosca_and_general', 'all').df
test = Selection(df, 'hoi')

# for dat in ['all', 'industry', 'example', 'a4c', 'forge', 'puccini']:
#     for met in ['general', 'tosca', 'tosca_and_general']:
#         Data(met, dat) 