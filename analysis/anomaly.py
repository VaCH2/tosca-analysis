from utils import scale_df
from data import Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pyod.models.cblof import CBLOF 
from pyod.models.feature_bagging import FeatureBagging 
from pyod.models.hbos import HBOS 
from pyod.models.iforest import IForest 
from pyod.models.knn import KNN 
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM

from pyod.utils.utility import standardizer
from pyod.models.combination import maximization
import matplotlib.pyplot as plt

class AnomalyDetector():

    def __init__(self, data, cutoff=3):
        if isinstance(data, Data):
            self.df = data.df
        elif isinstance(data, pd.DataFrame):
            self.df = data
            #self.df = self.df.drop('label', axis=1)

        score_df = self.df
        score_df['outlier_score'] = self.__train_classifiers()
        self.outliers = self.__identify_outliers(cutoff, score_df['outlier_score'])

    def __load_classifiers(self):
        outliers_fraction = 0.05
        random_state = np.random.RandomState(0)

        classifiers = {     
            'Cluster-based Local Outlier Factor (CBLOF)':
                CBLOF(contamination=outliers_fraction,
                    check_estimator=False, random_state=random_state),
            'Feature Bagging':
                FeatureBagging(LOF(n_neighbors=35),
                            contamination=outliers_fraction,
                            random_state=random_state),
            'Histogram-base Outlier Detection (HBOS)': HBOS(
                contamination=outliers_fraction),
            'Isolation Forest': IForest(contamination=outliers_fraction,
                                        random_state=random_state),
            'K Nearest Neighbors (KNN)': KNN(
                contamination=outliers_fraction),
            'Average KNN': KNN(method='mean',
                            contamination=outliers_fraction),
            'Local Outlier Factor (LOF)':
                LOF(n_neighbors=35, contamination=outliers_fraction),
            'Minimum Covariance Determinant (MCD)': MCD(
                contamination=outliers_fraction, random_state=random_state),
            'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        }

        return classifiers

    def __train_classifiers(self):
        scaler = MinMaxScaler(feature_range=(0,1))
        X = scaler.fit_transform(self.df.copy())
        classifiers = self.__load_classifiers()
        scores = np.zeros([X.shape[0], len(classifiers)])
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            try:
                clf.fit(X)
                scores[:, i] = clf.decision_scores_
            except Exception as e:
                print("Failed for ", clf_name)
                print("because of ", e)

            
        standard_scores = standardizer(scores)
        combined_scores = maximization(standard_scores)
        return combined_scores

    def __identify_outliers(self, cutoff, score_df):
        outliers = self.df[score_df > cutoff]
        return outliers
