import pandas as pd
import numpy as np
from data import Data
from anomaly import AnomalyDetector
from stats import Stats

data = Data('tosca_and_general', 'all')
rep_data = Data('tosca_and_general', 'repos')
anomalies = AnomalyDetector(rep_data)

def get_stats(datasets):
    mean_df = pd.DataFrame()
    nonzero_df = pd.DataFrame()
    min_df = pd.DataFrame()
    max_df = pd.DataFrame()
    for dataset in datasets:
        mean_df = pd.concat([mean_df, dataset.mean], axis=1, sort=False)
        nonzero_df = pd.concat([nonzero_df, dataset.nonzero], axis=1, sort=False)
        min_df = pd.concat([min_df, dataset.min], axis=1, sort=False)
        max_df = pd.concat([max_df, dataset.max], axis=1, sort=False)

    total_df = pd.concat([mean_df, nonzero_df, min_df, max_df], axis=1, sort=False)

    return total_df

datasets = [Stats(anomalies.outliers)]
results = get_stats(datasets)

datasets2 = [Stats(data)]
results2 = get_stats(datasets2)

#Stats laten duidelijk zien dat ook de anomaly detection focuessed op de grote scripts.

#%%

def check_repo(df):
    count = {'a4c' : 0, 'forge' : 0, 'puccini' : 0}
    a4c = df.index.str.contains('A4C')
    count['a4c'] = np.sum(a4c)
    forge = df.index.str.contains('Forge')
    count['forge'] = np.sum(forge)
    puc = df.index.str.contains('Puccini')
    count['puccini'] = np.sum(puc)
 
    return count

def check_purpose(df):
    count = {'industry' : 0, 'example' : 0}
    industry = df.index.str.contains('Total Industry')
    count['industry'] = np.sum(industry)
    example = df.index.str.contains('Total Examples')
    count['example'] = np.sum(example)

    return count

check_repo(anomalies.outliers)
purp_anomalies = AnomalyDetector(data, cutoff=2).outliers
check_purpose(purp_anomalies)

purp_anomalies['ttb_check'].value_counts()
#In alle 3 de gevallen zie je dat er geen onderscheid wordt gemaakt op de splits die wij voor ogen hadden. 
#Anomalies zijn gelijk verdeeld. Dus dit laat zien dat er geen verschil is op basis van anomalies. 



