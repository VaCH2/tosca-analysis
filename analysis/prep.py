import pandas as pd
import numpy as np
from data import Data
import pickle
from scipy.stats import chi2_contingency as chi2
from scipy.spatial.distance import braycurtis

class Preprocessing():
    def __init__(self, data, constants=None, corr=None, pca=False, anomalies=False, chi=False):
        '''For constants and corr please provide the list of to be dropped columns'''

        if isinstance(data, Data):
            self.df = data.df.head(10)
        elif isinstance(data, pd.DataFrame):
            self.df = data.head(10)

        self.anomalypercentage = None
        to_drop = []
        if corr != None:
            to_drop.extend(corr)

        if constants != None:
            to_drop.extend(constants)
        
        to_drop = list(set(to_drop))
        
        self.df = self.df.drop(to_drop, axis=1)
        
        if anomalies != None:
            self.anomalypercentage = self.filter_anomalies()

        if chi:
            self.df = self.__transform_distance()


    def __transform_distance(self):
        matrix = pd.DataFrame(index=self.df.index, columns=self.df.index)
        # Werkt niet omdat de data veel te sparse is.. Stond ergens een minimale count van 5 per kolom. 
        #Nu maar braycurtis genomen maar weet niet of hier wat haken en ogen aan zitten..
        combs = [ ((x,y), braycurtis(self.df.loc[[x], :].to_numpy(),self.df.loc[[y], :].to_numpy())) for x in list(self.df.index) for y in list(self.df.index)]
        for element in combs:
            matrix.loc[element[0][0], element[0][1]] = element[1]
        return matrix




    def filter_anomalies(self):
        return 0


data = Data('tosca_and_general', 'all')
prep = Preprocessing(data, chi=True)
