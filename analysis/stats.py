import pandas as pd
import numpy as np
from data import Data
import pickle


class Stats():
    def __init__(self, data):
        '''Enter dataclass of pandas dataframe'''
        
        if isinstance(data, Data):
            self.df = data.df
        elif isinstance(data, pd.DataFrame):
            self.df = data

        self.totalsparsity = self.calc_sparsity()
        self.featuresparsity = self.calc_featuresparsity()
        self.constants = self.constantvalues()
        self.corrfeatures = self.correlation()
        self.mean = self.calc_mean()
        self.nonzero = self.calc_nonzero()
        self.zero = self.calc_zero()
        self.min = self.calc_min()
        self.max = self.calc_max()
        self.stddv = self.calc_stddv()
        self.q1 = self.calc_q1()
        self.median = self.calc_median()
        self.q3 = self.calc_q3()


    def calc_sparsity(self):
        '''Calculate the sparsity of the selected data'''
        zeroes = 0
        for column in self.df.columns:
            zeroes += np.count_nonzero(self.df[column] == 0)
        return zeroes / (self.df.shape[0] * self.df.shape[1])
    
    def calc_featuresparsity(self):
        '''Calculate sparsity per feature'''
        df = self.df
        result = pd.DataFrame()
        result['sparsity'] = df.apply(lambda x: np.count_nonzero(x == 0)/len(x))
        return result


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
        result['nonzero'] = df.apply(lambda x: (np.count_nonzero(x)))
        return result

    def calc_zero(self):
        df = self.df
        result = pd.DataFrame()
        result['zero'] = df.apply(lambda x: (np.count_nonzero(x == 0)))
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

    def calc_stddv(self):
        df = self.df
        result = pd.DataFrame()
        result['stddv'] = df.apply(lambda x: np.std(x))
        return result


    def calc_q1(self):
        df = self.df
        result = pd.DataFrame()
        result['q1'] = df.apply(lambda x: np.quantile(x, 0.25))
        return result


    def calc_median(self):
        df = self.df
        result = pd.DataFrame()
        result['median'] = df.apply(lambda x: np.quantile(x, 0.5))
        return result
    
    def calc_q3(self):
        df = self.df
        result = pd.DataFrame()
        result['q3'] = df.apply(lambda x: np.quantile(x, 0.75))
        return result