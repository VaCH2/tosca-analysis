import pandas as pd
import numpy as np
from data import Data
import pickle

class Preprocessing():
    def __init__(self, df, corr=False, pca=False):
        self.df = df
        if corr == True:
            self.df = self.correlation()
        if pca == True:
            self.df = self.pca()

    
    def correlation(self):
        '''Drop the high correlation variables, and store the high correlation values.'''
        corr = self.df.corr(method='pearson').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        high_correlations = [column for column in upper.columns if any(upper[column] > 0.90)]
        df = self.df.drop(self.df[high_correlations], axis=1)
        return df

    def pca(self):
        pass

    def ietsmetresultaten(self):
        pass

df = Data('tosca_and_general', 'all').df
test = Preprocessing(df, 'hoi')