import pandas as pd
import numpy as np
from data import Data
import pickle

class Stats():
    def __init__(self, df):
        self.df = df
        self.sparsity = self.calc_sparsity()
        self.constants = self.constantvalues()
        self.corrfeatures = self.correlation()

    def calc_sparsity(self):
        '''Calculate the sparsity of the selected data'''
        zeroes = 0
        for column in self.df.columns:
            zeroes += np.count_nonzero(self.df[column] == 0)
        return zeroes / (self.df.shape[0] * self.df.shape[1])

    def constantvalues(self):
        '''Drop the variables which have a contant value'''
        constant_columns = [column for column in self.df.columns if len(self.df[column].unique()) < 2]
        self.df = self.df.drop(self.df[constant_columns], axis=1)
        return constant_columns
    
    def correlation(self):
        '''Drop the high correlation variables, and store the high correlation values.'''
        corr = self.df.corr(method='pearson').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        high_correlations = [column for column in upper.columns if any(upper[column] > 0.90)]
        return high_correlations

    def pca(self):
        pass

    def ietsmetresultaten(self):
        pass

df = Data('tosca_and_general', 'all', 'topology')
#test = Preprocessing(df, 'hoi')