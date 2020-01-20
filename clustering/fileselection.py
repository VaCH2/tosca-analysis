import pandas as pd
#from data.data import Data
import pickle

class FileSelection():
    def __init__(self, df, criteria):
        self.df = df


    def correlation(self):
        corr = self.df.corr(method='pearson')
        return corr



df = pickle.load(open('../temp_data/{}_{}_raw_df'.format('tosca_and_general', 'all'), 'rb'))
#df = Data('tosca_and_general', 'all').df
test = FileSelection(df, 'hoi')