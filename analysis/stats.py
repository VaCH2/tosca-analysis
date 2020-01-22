import pandas as pd
import numpy as np
from data import Data
import pickle

class Stats():
    def __init__(self, data):
        self.df = data.df
        self.sparsity = self.calc_sparsity()
        self.constants = self.constantvalues()
        self.corrfeatures = self.correlation()
        self.descriptives = self.df.describe()

    def calc_sparsity(self):
        '''Calculate the sparsity of the selected data'''
        zeroes = 0
        for column in self.df.columns:
            zeroes += np.count_nonzero(self.df[column] == 0)
        return zeroes / (self.df.shape[0] * self.df.shape[1])

    def constantvalues(self):
        '''Collect the variables which have a contant value'''
        constant_columns = [column for column in self.df.columns if len(self.df[column].unique()) < 2]
        return constant_columns
    
    def correlation(self):
        '''Collect the high correlation variables (> 0.90)'''
        corr = self.df.corr(method='pearson').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        high_correlations = [column for column in upper.columns if any(upper[column] > 0.90)]
        return high_correlations


df = Data('tosca_and_general', 'all', 'topology')
#test = Preprocessing(df, 'hoi')