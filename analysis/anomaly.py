from utils import scale_df
from data import Data
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pyod.models.cblof import CBLOF 
from pyod.models.feature_bagging import FeatureBagging 
from pyod.models.hbos import HBOS 
from pyod.models.iforest import IForest 
from pyod.models.knn import KNN 
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM

class AnomalyDetector():

    def __init__(self, data):
        self.df = data.df
        self.scores = self.__train_classifiers()

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

    #Ze failen hier allemaal? Wat gaat er mis?
    def __train_classifiers(self):
        scaler = MinMaxScaler(feature_range=(0,1))
        X = scaler.fit_transform(self.df.copy())
        print(X)
        classifiers = self.__load_classifiers()
        scores = np.zeros([X.shape[0], len(classifiers)])
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            try:
                clf.fit(X)
                scores[:, clf_name] = clf.decision_scores_
            except Exception as e:
                print("Failed for ", clf_name)
                print("because of ", e)
        return scores

data = Data('tosca_and_general', 'all')
anomaly = AnomalyDetector(data)

    