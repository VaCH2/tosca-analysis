import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

class Significance():
    def __init__(self, data1, data2, discardzeroes=False):
        self.data1 = data1
        self.data2 = data2

        if not list(self.data1.columns) == list(self.data2.columns):
            raise ValueError('datasets do not contain the same columns')

        if len(self.data1) < 20 or len(self.data2) < 20:
            #print('One of the samples contained less than 20 data points!')
            self.rejected_features = pd.DataFrame()

        else:
            print('Both samples contained more than 20 data points! Total: ')
            uncorrected_p_values = self.__calc_sig(discardzeroes)
            self.uncorrected_p_values = uncorrected_p_values.dropna() 
            self.sig = self.__multitest_correction(self.uncorrected_p_values, 0.01)
            self.rejected_features = self.sig[self.sig['rejected'] == True]

    
    def __calc_sig(self, discardzeroes):
        '''Calculates the p value for all the features based on the U-test
        and returns a list of p-values'''
        
        columns = self.data1.columns
        p_values = pd.DataFrame(columns=['p_values'])

        for column in columns:
            try:
                if discardzeroes:
                    df1, df2 = self.__discard_zeroes_and_balance(self.data1, self.data2, column)
                    if len(df1) < 20 or len(df2) < 20:
                        continue                    
                    stat, p = mannwhitneyu(df1[column], df2[column])
                else:
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

    def __discard_zeroes_and_balance(self, df1, df2, column):
        df1 = df1[df1[column] != 0]
        df2 = df2[df2[column] != 0]
        # len_df1 = df1.shape[0]
        # len_df2 = df2.shape[0]

        # #downsampling
        # if len_df1 < len_df2:
        #     df2.sample(n=len_df1, replace=True, random_state=1)

        # elif len_df2 < len_df1:
        #     df1.sample(n=len_df2, replace=True, random_state=1)
        
        # else:
        #     pass
            
        return df1, df2