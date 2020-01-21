import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

class Significance():
    
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

        if not list(self.data1.columns) == list(self.data2.columns):
            raise ValueError('datasets do not contain the same columns')

        uncorrected_p_values = self.__calc_sig()
        self.uncorrected_p_values = uncorrected_p_values.dropna() 
        self.sig = self.__multitest_correction(self.uncorrected_p_values, 0.05)
        self.rejected_features = self.sig[self.sig['rejected'] == True]

    
    def __calc_sig(self):
        '''Calculates the p value for all the features based on the TO DETERMINE test
        and returns a list of p-values. Each sample must contain at least 20 data points'''

        if len(self.data1) < 20 or len(self.data2) < 20:
            print('One of the samples contained less than 20 data points!')
        
        columns = self.data1.columns
        p_values = pd.DataFrame(columns=['p_values'])

        for column in columns:
            try:
                stat, p = mannwhitneyu(self.data1[column], self.data2[column])

                value = pd.Series(data={'p_values' : p}, name=column)
                p_values = p_values.append(value, ignore_index=False)
            except ValueError:
                value = pd.Series(data={'p_values' : np.nan}, name=column)
                p_values = p_values.append(value, ignore_index=False)

        return p_values

    def __multitest_correction(self, p_values, alpha):
        '''A correction based on the Benjamini/Hochberg procedure for the multiple comparison problem.
        Returns a corrected p-value per hypothesis and a rejection boolean based on the provided alpha'''
        values = p_values['p_values'].tolist()
        testcorrection = multipletests(values, alpha, 'fdr_bh')
        d = {'rejected' : testcorrection[0], 'corr_p_values' : testcorrection[1]}
        p_values = pd.DataFrame(data=d, index=self.uncorrected_p_values.index)
        return p_values