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
        self.mean = self.calc_mean()
        self.nonzero = self.calc_nonzero()
        self.min = self.calc_min()
        self.max = self.calc_max()


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

    def calc_mean(self):
        df = self.df
        result = pd.DataFrame()
        result['mean'] = df.apply(lambda x: np.mean(x))
        return result

    def calc_nonzero(self):
        df = self.df
        result = pd.DataFrame()
        result['% nonzero'] = df.apply(lambda x: (np.count_nonzero(x) / len(df.index)))
        return result

    def calc_min(self):
        df = self.df
        result = pd.DataFrame()
        result['min'] = df.apply(lambda x: np.min(x))
        return result

    def calc_max(self):
        df = self.df
        result = pd.DataFrame()
        result['max'] = df.apply(lambda x: np.max(x))
        return result


#df = Data('tosca_and_general', 'all', 'topology')
#test = Preprocessing(df, 'hoi')