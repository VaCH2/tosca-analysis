import pickle
import os
import csv

from data import Data
from stats import Stats
from anomaly import AnomalyDetector
from utils import scale_df

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score


root_folder = os.path.dirname(os.path.dirname( __file__ ))
print(root_folder)
results_folder = os.path.join(root_folder, 'results', 'clustering_models')


def main():
    smells_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'smells_df'), 'rb'))
    smells_df = smells_df.drop(r'SeaCloudsEU\tosca-parser\Industry\normative_types.yaml')
    smells_df = smells_df.astype(bool)

    df = Data().dfs.get('all')
    df = df.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)

    for smell in smells_df.columns:
        for option in get_options():
            option_df = calculate_option(smell, df, smells_df, option[0], option[1], option[2], option[3], option[4])
            scores = calculate_internal_measurements(smell, option_df)

            option_name = f'excludeoutliers-{option[0]}_excludecorr-{option[1]}_pca-{option[2]}_{option[3]}'
            
            if len(option[4]) == 2:
                option_name = f'{option_name}_{option[4][0]}_{option[4][1]}'
            else:
                option_name = f'{option_name}_{option[4]}'

            scores.insert(0, option_name)

            with open(os.path.join(results_folder, f'clustering_scores_{smell}.csv'), mode='a', newline='') as file_:
                writer = csv.writer(file_)
                writer.writerow(scores)






def handle_correlations(df):
    matrix = df.corr()
    matrix = matrix.where(~np.tril(np.ones(matrix.shape)).astype(np.bool))
    matrix = matrix.stack()
    matrix = matrix[matrix > 0.8]
    matrix = matrix.to_frame()
    first_level = list(matrix.index.get_level_values(0))
    df = df.drop(list(set(first_level)), axis=1)
    return df

def handle_pca(df):
    pca = PCA(n_components=20)
    X_std = scale_df(df)
    principalComponents = pca.fit_transform(X_std)
    df = pd.DataFrame(principalComponents, index=df.index.values)
    return df

def handle_anomalies(df):
    outliers = AnomalyDetector(df).outliers
    df = df.drop(outliers.index.values, axis=0)
    return df

def balance_smell(smell, df, smells_df):
    smells_df = smells_df.loc[df.index.values]

    true_df = smells_df[smells_df[smell] == True]
    false_df = smells_df[smells_df[smell] == False]
    true_len = true_df.shape[0]
    false_len = false_df.shape[0]
    
    #downsampling
    if true_len < false_len:
        false_df = false_df.sample(n=true_len, replace=True, random_state=1)

    elif false_len < true_len:
        true_df = true_df.sample(n=false_len, replace=True, random_state=1)
    
    else:
        pass

    ix_list = list(true_df.index.values) + list(false_df.index.values)
    balanced_df = df.loc[ix_list]
    return balanced_df, smells_df[smell]


def handle_distance_function(dist_function, df):
    X_std = scale_df(df)
    distances = pairwise_distances(X_std, metric=dist_function)
    distances = pd.DataFrame(data=distances, index=df.index, columns=df.index)
    return distances

def handle_algorithm(algorithm, distancematrix, df):
    if algorithm[0] == 'agglo':
        model = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage=algorithm[1])
        labels = model.fit_predict(distancematrix.to_numpy())
    if algorithm == 'dbscan':
        model = DBSCAN(metric='precomputed')
        labels = model.fit_predict(distancematrix.to_numpy())
        
    if algorithm[0] == 'gm':
        model = GaussianMixture(n_components=2, covariance_type=algorithm[1])
        labels = model.fit_predict(df.to_numpy())
    
    df['cluster'] = labels
    return df


def calculate_internal_measurements(smell, df):

    y_pred = df['cluster']
    y_true = df[smell]
    x = df.drop([smell, 'cluster'], axis=1)
    try:
        sil_score = silhouette_score(X=x.to_numpy(), labels=y_pred)
    except:
        sil_score = np.nan
    try:
        dav_score = davies_bouldin_score(X=x.to_numpy(), labels=y_pred)
    except:
        dav_score = np.nan
    try:
        ch_score = calinski_harabasz_score(X=x.to_numpy(), labels=y_pred)
    except:
        ch_score = np.nan
    try:
        ari_score = adjusted_rand_score(y_true, y_pred)
    except:
        ari_score = np.nan
    try:
        ami_score = adjusted_mutual_info_score(y_true, y_pred)
    except:
        ami_score = np.nan
    try:
        nmi_score = normalized_mutual_info_score(y_true, y_pred)
    except:
        nmi_score = np.nan
    try:
        homogen_score = homogeneity_score(y_true, y_pred)
    except:
        homogen_score = np.nan
    try:
        complete_score = completeness_score(y_true, y_pred)
    except:
        complete_score = np.nan

    #V_measure is a combination of homogeneity and completeness
    try:
        v_score = v_measure_score(y_true, y_pred)
    except:
        v_score = np.nan
    try:
        fm_score = fowlkes_mallows_score(y_true, y_pred)
    except:
        fm_score = np.nan

    return [sil_score, dav_score, ch_score, ari_score, ami_score, nmi_score, homogen_score, complete_score, v_score, fm_score]



def calculate_option(smell, df, smells_df, exclude_outliers, exclude_corr, pca, distance, algorithm):
    try:
        df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'clustering', f'clusteringpipeline_smell-{smell}_excludeoutliers-{exclude_outliers}_excludecorr-{exclude_corr}_pca-{pca}_{distance}_{algorithm}'), 'rb'))

    except (OSError, IOError):
        if exclude_corr:
            df = handle_correlations(df)

        if distance != None:
            try:
                distance_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'clustering', f'distance_excludecorr-{exclude_corr}_{distance}'), 'rb'))
            except (OSError, IOError):
                distance_df = handle_distance_function(distance, df)
                pickle.dump(distance_df, open(os.path.join(root_folder, 'temp_data', 'clustering', f'distance_excludecorr-{exclude_corr}_{distance}'), 'wb'))

        if exclude_outliers:
            df = handle_anomalies(df)
            df = df.drop(columns='outlier_score')

        if pca:
            df = handle_pca(df)
            if distance != None:
                distance_df = handle_distance_function(distance, df)

        x_df, y_df = balance_smell(smell, df, smells_df)
 
        #Filter only applicable distances out
        if distance != None:
            distance_df = distance_df.loc[x_df.index, x_df.index]
        else:
            distance_df = None

        
        x_df = handle_algorithm(algorithm, distance_df, x_df)
        df = pd.merge(x_df, y_df, right_index=True, left_index=True)
        pickle.dump(df, open(os.path.join(root_folder, 'temp_data', 'clustering', f'clusteringpipeline_smell-{smell}_excludeoutliers-{exclude_outliers}_excludecorr-{exclude_corr}_pca-{pca}_{distance}_{algorithm}'), 'wb'))
    
    #Fix hot encoding
    count_df = pd.DataFrame(index=[True, False])
    for label in df['cluster'].unique():
        s = df[df['cluster'] == label][smell].value_counts()
        count_df[label] = s

    highest_true = count_df.idxmax(axis=1)[True]
    highest_false = count_df.idxmax(axis=1)[False]
    if highest_true == highest_false:
        need_second_best = count_df[highest_false].idxmin()
        if need_second_best == True:
            highest_true = count_df.loc[True].nlargest(2).idxmin()
        else:
            highest_false = count_df.loc[False].nlargest(2).idxmin()
    
    df[smell] = df[smell].map({True: highest_true, False: highest_false})  


    return df
    


def get_options():
    options = [
        (False, False, True, None, ('gm', 'full') ),
        (False, True, True, None, ('gm', 'full')  ),
        (True, False, True, None, ('gm', 'full')  ),
        (True, True, True, None, ('gm', 'full')  ),
        (False, False, True, None, ('gm', 'tied')  ),
        (False, True, True, None, ('gm', 'tied') ),
        (True, False, True, None, ('gm', 'tied') ),
        (True, True, True, None, ('gm', 'tied') ),
        (False, False, True, None, ('gm', 'diag') ),
        (False, True, True, None, ('gm', 'diag') ),
        (True, False, True, None, ('gm', 'diag') ),
        (True, True, True, None, ('gm', 'diag') ),
        (False, False, True, None, ('gm', 'spherical') ),
        (False, True, True, None, ('gm', 'spherical') ),
        (True, False, True, None, ('gm', 'spherical') ),
        (True, True, True, None, ('gm', 'spherical') ),
        (False, False, False, None, ('gm', 'full') ),
        (False, True, False, None, ('gm', 'full')  ),
        (True, False, False, None, ('gm', 'full')  ),
        (True, True, False, None, ('gm', 'full')  ),
        (False, False, False, None, ('gm', 'tied')  ),
        (False, True, False, None, ('gm', 'tied') ),
        (True, False, False, None, ('gm', 'tied') ),
        (True, True, False, None, ('gm', 'tied') ),
        (False, False, False, None, ('gm', 'diag') ),
        (False, True, False, None, ('gm', 'diag') ),
        (True, False, False, None, ('gm', 'diag') ),
        (True, True, False, None, ('gm', 'diag') ),
        (False, False, False, None, ('gm', 'spherical') ),
        (False, True, False, None, ('gm', 'spherical') ),
        (True, False, False, None, ('gm', 'spherical') ),
        (True, True, False, None, ('gm', 'spherical') ),

        (False, False, False, 'cosine', ('agglo', 'complete') ),
        (False, True, False, 'cosine', ('agglo', 'complete') ),
        (True, False, False, 'cosine', ('agglo', 'complete') ),
        (True, True,False, 'cosine', ('agglo', 'complete') ),
        (False, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, 'l1', ('agglo', 'single') ),
        (False, True, False, 'l1', ('agglo', 'single') ),
        (True, False, False, 'l1', ('agglo', 'single') ),
        (True, True, False, 'l1', ('agglo', 'single') ),
        (False, False, True, 'cosine', ('agglo', 'complete') ),
        (False, True, True, 'cosine', ('agglo', 'complete') ),
        (True, False, True, 'cosine', ('agglo', 'complete') ),
        (True, True, True, 'cosine', ('agglo', 'complete') ),
        (False, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, 'l1', ('agglo', 'single') ),
        (False, True, True, 'l1', ('agglo', 'single') ),
        (True, False, True, 'l1', ('agglo', 'single') ),
        (True, True, True, 'l1', ('agglo', 'single') ),
        (False, False, False, 'cosine', ('agglo', 'complete') ),
        (False, True, False, 'cosine', ('agglo', 'complete') ),
        (True, False, False, 'cosine', ('agglo', 'complete') ),
        (True, True, False, 'cosine', ('agglo', 'complete') ),
        (False, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, 'l1', ('agglo', 'single') ),
        (False, True, False, 'l1', ('agglo', 'single') ),
        (True, False, False, 'l1', ('agglo', 'single') ),
        (True, True, False, 'l1', ('agglo', 'single') ),
        (False, False, True, 'cosine', ('agglo', 'complete') ),
        (False, True, True, 'cosine', ('agglo', 'complete') ),
        (True, False, True, 'cosine', ('agglo', 'complete') ),
        (True, True, True, 'cosine', ('agglo', 'complete') ),
        (False, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, 'l1', ('agglo', 'single') ),
        (False, True, True, 'l1', ('agglo', 'single') ),
        (True, False, True, 'l1', ('agglo', 'single') ),
        (True, True, True, 'l1', ('agglo', 'single') ),
        (False, False, False, 'cosine', ('agglo', 'complete') ),
        (False, True, False, 'cosine', ('agglo', 'complete') ),
        (True, False, False, 'cosine', ('agglo', 'complete') ),
        (True, True, False, 'cosine', ('agglo', 'complete') ),
        (False, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, 'l1', ('agglo', 'single') ),
        (False, True, False, 'l1', ('agglo', 'single') ),
        (True, False, False, 'l1', ('agglo', 'single') ),
        (True, True, False, 'l1', ('agglo', 'single') ),
        (False, False, True, 'cosine', ('agglo', 'complete') ),
        (False, True, True, 'cosine', ('agglo', 'complete') ),
        (True, False, True, 'cosine', ('agglo', 'complete') ),
        (True, True, True, 'cosine', ('agglo', 'complete') ),
        (False, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, 'l1', ('agglo', 'single') ),
        (False, True, True, 'l1', ('agglo', 'single') ),
        (True, False, True, 'l1', ('agglo', 'single') ),
        (True, True, True, 'l1', ('agglo', 'single') ),


        (False, False, False, 'cosine', 'dbscan' ),
        (False, True, False, 'cosine', 'dbscan' ),
        (True, False, False, 'cosine', 'dbscan' ),
        (True, True, False, 'cosine', 'dbscan' ),
        (False, False, False, 'braycurtis', 'dbscan' ),
        (False, True, False, 'braycurtis', 'dbscan' ),
        (True, False, False, 'braycurtis', 'dbscan' ),
        (True, True, False, 'braycurtis', 'dbscan' ),
        (False, False, False, 'l1', 'dbscan' ),
        (False, True, False, 'l1', 'dbscan' ),
        (True, False, False, 'l1', 'dbscan' ),
        (True, True, False, 'l1', 'dbscan' ),
        (False, False, True, 'cosine', 'dbscan' ),
        (False, True, True, 'cosine', 'dbscan' ),
        (True, False, True, 'cosine', 'dbscan' ),
        (True, True, True, 'cosine', 'dbscan' ),
        (False, False, True, 'braycurtis', 'dbscan' ),
        (False, True, True, 'braycurtis', 'dbscan' ),
        (True, False, True, 'braycurtis', 'dbscan' ),
        (True, True, True, 'braycurtis', 'dbscan' ),
        (False, False, True, 'l1', 'dbscan' ),
        (False, True, True, 'l1', 'dbscan' ),
        (True, False, True, 'l1', 'dbscan' ),
        (True, True, True, 'l1', 'dbscan' )
    ]
    return options

if __name__ == '__main__':
    main()


# smell='db'
# df=None
# smells_df=None
# exclude_outliers=True
# exclude_corr=False
# pca=False
# algorithm=('gm', 'spherical')
# distance=None
# df = calculate_option(smell, df, smells_df, exclude_outliers, exclude_corr, pca, distance, algorithm)

# from significance import Significance

# df0 = df[df['cluster'] == 0]
# df1 = df[df['cluster'] == 1]
# sig_analysis = Significance(df0, df1)