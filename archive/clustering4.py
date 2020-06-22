# Class om de shit uit te rekenen
# Main function met intializatie om configurations uit te rekenen en complete class instance op te slaan in pickle
# constant drop altijd doen!

#Configclass:
# input: config tuple
# NO PICKLE LOOKUP!
# do preprocessing (also constant drop!), cluster calculation, AND performance score calculation
# object.params 
# object.model (cluster model)
# object.scores (df met scores ofzo)
# object.stability (nog ff kijken hoe dit past)
# object.name (Dit moet de index van het model zijn)


import pickle
import os
import csv

from data import Data
from utils import scale_df

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

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
results_folder = os.path.join(root_folder, 'results', 'clustering_models')
smells_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'smells_df'), 'rb'))

def main():
    smells_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'smells_df'), 'rb'))
    
    #Add dummy smell to calculate the total cluster
    smells_df['alldummy'] = pd.Series([True for x in range(552)] + [False for x in range(552)], index=smells_df.index)
    
    smells_df = smells_df.drop(r'SeaCloudsEU\tosca-parser\Industry\normative_types.yaml')
    smells_df = smells_df.astype(bool)

    df = Data().dfs.get('all')
    df = df.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
    
    for smell in smells_df.columns:

        with open(os.path.join(results_folder, f'clustering_scores_{smell}.csv'), mode='a', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerow(['name', 'sil_score', 'dav_score', 'ch_score', 'ari_score', 'ami_score', 'nmi_score', 'homogen_score', 'complete_score', 'v_score', 'fm_score'])


        for option in get_options():
            if smell is 'alldummy':
                for k in [2,3,4,5,6,7]:
                    try:
                        print(option)
                        option_df = calculate_option(smell, df, smells_df, option[0], option[1], option[2], option[3], k, option[4], option[5])
                        scores = calculate_internal_measurements(smell, option_df)

                        option_name = f'excludeoutliers-{option[0]}_excludecorr-{option[1]}_pca-{option[2]}_excludespars-{option[3]}_{k}_{option[4]}'
                        
                        if len(option[5]) == 2:
                            option_name = f'{option_name}_{option[5][0]}_{option[5][1]}'
                        else:
                            option_name = f'{option_name}_{option[5]}'

                        scores.insert(0, option_name)

                        with open(os.path.join(results_folder, f'clustering_scores_{smell}.csv'), mode='a', newline='') as file_:
                            writer = csv.writer(file_)
                            writer.writerow(scores)
                    except:
                        pass
            
            else:
                try:
                    print(option)
                    option_df = calculate_option(smell, df, smells_df, option[0], option[1], option[2], option[3], 2, option[4], option[5])
                    scores = calculate_internal_measurements(smell, option_df)

                    option_name = f'excludeoutliers-{option[0]}_excludecorr-{option[1]}_pca-{option[2]}_excludespars-{option[3]}_2_{option[4]}'
                    
                    if len(option[5]) == 2:
                        option_name = f'{option_name}_{option[5][0]}_{option[5][1]}'
                    else:
                        option_name = f'{option_name}_{option[5]}'

                    scores.insert(0, option_name)

                    with open(os.path.join(results_folder, f'clustering_scores_{smell}.csv'), mode='a', newline='') as file_:
                        writer = csv.writer(file_)
                        writer.writerow(scores)
                
                except:
                    pass





def handle_correlations(df):
    matrix = df.corr()
    matrix = matrix.where(~np.tril(np.ones(matrix.shape)).astype(np.bool))
    matrix = matrix.stack()
    matrix = matrix[matrix > 0.8]
    matrix = matrix.to_frame()
    first_level = list(matrix.index.get_level_values(0))
    df = df.drop(list(set(first_level)), axis=1)
    return df

def handle_sparsity(df):
    sparsity = Stats(df).featuresparsity
    sparse_measurements = sparsity[sparsity['sparsity'] > 0.9].index.values
    df = df.drop(sparse_measurements, axis=1)
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

def handle_algorithm(algorithm, distancematrix, df, n):
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n)
        labels = model.fit_predict(df.to_numpy())
    if algorithm == 'specclu':
        if distancematrix is None:
            model = SpectralClustering(n_clusters=n)
            labels = model.fit_predict(df.to_numpy())
        else:
            model = SpectralClustering(n_clusters=n, affinity='precomputed')
            labels = model.fit_predict(distancematrix.to_numpy())

    if algorithm[0] == 'agglo':
        model = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage=algorithm[1])
        labels = model.fit_predict(distancematrix.to_numpy())
    if algorithm == 'dbscan':
        model = DBSCAN(metric='precomputed')
        labels = model.fit_predict(distancematrix.to_numpy())
        
    if algorithm[0] == 'gm':
        model = GaussianMixture(n_components=n, covariance_type=algorithm[1])
        labels = model.fit_predict(df.to_numpy())
    
    df['cluster'] = labels
    return df


def calculate_internal_measurements(smell, df):

    y_pred = df['cluster']

    if y_pred.dtype == bool:
        y_pred = y_pred.map({True: 1, False: 0})

    y_true = df[smell]
    y_true = y_true.map({True: 1, False: 0})

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
    
    if df[smell].name != 'alldummy':
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

    else:
        return [sil_score, dav_score, ch_score]



def calculate_option(smell, df, smells_df, exclude_outliers, exclude_corr, pca, exclude_spars, n, distance, algorithm):

    if isinstance(algorithm, tuple):
        algo_name = f'{algorithm[0]}_{algorithm[1]}'
    else:
        algo_name = algorithm


    try:
        df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'clustering', f'clusteringpipeline_smell-{smell}_excludeoutliers-{exclude_outliers}_excludecorr-{exclude_corr}_pca-{pca}_excludespars-{exclude_spars}_{n}_{distance}_{algo_name}'), 'rb'))

    except (OSError, IOError):
        if exclude_corr:
            df = handle_correlations(df)

        if exclude_spars:
            df = handle_sparsity(df)

        if distance != None:
            try:
                distance_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'clustering', f'distance_excludecorr-{exclude_corr}_excludespars-{exclude_spars}_{distance}'), 'rb'))
            except (OSError, IOError):
                distance_df = handle_distance_function(distance, df)
                pickle.dump(distance_df, open(os.path.join(root_folder, 'temp_data', 'clustering', f'distance_excludecorr-{exclude_corr}_excludespars-{exclude_spars}_{distance}'), 'wb'))

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

        x_df = handle_algorithm(algorithm, distance_df, x_df, n)
        df = pd.merge(x_df, y_df, right_index=True, left_index=True)
            
        if df[smell].name != 'alldummy':
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
            
            #df[smell] = df[smell].map({True: highest_true, False: highest_false})
            df['cluster'] = df['cluster'].map({highest_true : True, highest_false : False})  

        pickle.dump(df, open(os.path.join(root_folder, 'temp_data', 'clustering', f'clusteringpipeline_smell-{smell}_excludeoutliers-{exclude_outliers}_excludecorr-{exclude_corr}_pca-{pca}_excludespars-{exclude_spars}_{n}_{distance}_{algo_name}'), 'wb'))

    return df
    


def get_options():
    options = [
        (False, False, True, True, None,  ('gm', 'full') ),
        (False, True, True, True, None,  ('gm', 'full')  ),
        (True, False, True, True, None,  ('gm', 'full')  ),
        (True, True, True, True, None,  ('gm', 'full')  ),
        (False, False, True, True, None,  ('gm', 'tied')  ),
        (False, True, True, True, None,  ('gm', 'tied') ),
        (True, False, True, True, None,  ('gm', 'tied') ),
        (True, True, True, True, None,  ('gm', 'tied') ),
        (False, False, True, True, None,  ('gm', 'diag') ),
        (False, True, True, True, None,  ('gm', 'diag') ),
        (True, False, True, True, None,  ('gm', 'diag') ),
        (True, True, True, True, None,  ('gm', 'diag') ),
        (False, False, True, True, None,  ('gm', 'spherical') ),
        (False, True, True, True, None,  ('gm', 'spherical') ),
        (True, False, True, True, None,  ('gm', 'spherical') ),
        (True, True, True, True, None,  ('gm', 'spherical') ),
        (False, False, False, True, None,  ('gm', 'full') ),
        (False, True, False, True, None,  ('gm', 'full')  ),
        (True, False, False, True, None,  ('gm', 'full')  ),
        (True, True, False, True, None,  ('gm', 'full')  ),
        (False, False, False, True, None,  ('gm', 'tied')  ),
        (False, True, False, True, None,  ('gm', 'tied') ),
        (True, False, False, True, None,  ('gm', 'tied') ),
        (True, True, False, True, None,  ('gm', 'tied') ),
        (False, False, False, True, None,  ('gm', 'diag') ),
        (False, True, False, True, None,  ('gm', 'diag') ),
        (True, False, False, True, None,  ('gm', 'diag') ),
        (True, True, False, True, None,  ('gm', 'diag') ),
        (False, False, False, True, None,  ('gm', 'spherical') ),
        (False, True, False, True, None,  ('gm', 'spherical') ),
        (True, False, False, True, None,  ('gm', 'spherical') ),
        (True, True, False, True, None,  ('gm', 'spherical') ),
        (False, False, True, False, None,  ('gm', 'full') ),
        (False, True, True, False, None,  ('gm', 'full')  ),
        (True, False, True, False, None,  ('gm', 'full')  ),
        (True, True, True, False, None,  ('gm', 'full')  ),
        (False, False, True, False, None,  ('gm', 'tied')  ),
        (False, True, True, False, None,  ('gm', 'tied') ),
        (True, False, True, False, None,  ('gm', 'tied') ),
        (True, True, True, False, None,  ('gm', 'tied') ),
        (False, False, True, False, None,  ('gm', 'diag') ),
        (False, True, True, False, None,  ('gm', 'diag') ),
        (True, False, True, False, None,  ('gm', 'diag') ),
        (True, True, True, False, None,  ('gm', 'diag') ),
        (False, False, True, False, None,  ('gm', 'spherical') ),
        (False, True, True, False, None,  ('gm', 'spherical') ),
        (True, False, True, False, None,  ('gm', 'spherical') ),
        (True, True, True, False, None,  ('gm', 'spherical') ),
        (False, False, False, False, None,  ('gm', 'full') ),
        (False, True, False, False, None,  ('gm', 'full')  ),
        (True, False, False, False, None,  ('gm', 'full')  ),
        (True, True, False, False, None,  ('gm', 'full')  ),
        (False, False, False, False, None,  ('gm', 'tied')  ),
        (False, True, False, False, None,  ('gm', 'tied') ),
        (True, False, False, False, None,  ('gm', 'tied') ),
        (True, True, False, False, None,  ('gm', 'tied') ),
        (False, False, False, False, None,  ('gm', 'diag') ),
        (False, True, False, False, None,  ('gm', 'diag') ),
        (True, False, False, False, None,  ('gm', 'diag') ),
        (True, True, False, False, None,  ('gm', 'diag') ),
        (False, False, False, False, None,  ('gm', 'spherical') ),
        (False, True, False, False, None,  ('gm', 'spherical') ),
        (True, False, False, False, None,  ('gm', 'spherical') ),
        (True, True, False, False, None,  ('gm', 'spherical') ),

        (False, False, False, False, None,  'kmeans' ),
        (False, True, False, False, None,  'kmeans' ),
        (True, False, False, False, None,  'kmeans' ),
        (True, True, False, False, None,  'kmeans' ),
        (False, False, True, False, None,  'kmeans' ),
        (False, True, True, False, None,  'kmeans' ),
        (True, False, True, False, None,  'kmeans' ),
        (True, True, True, False, None,  'kmeans' ),
        (False, False, False, True, None,  'kmeans' ),
        (False, True, False, True, None,  'kmeans' ),
        (True, False, False, True, None,  'kmeans' ),
        (True, True, False, True, None,  'kmeans' ),
        (False, False, True, True, None,  'kmeans' ),
        (False, True, True, True, None,  'kmeans' ),
        (True, False, True, True, None,  'kmeans' ),
        (True, True, True, True, None,  'kmeans' ),

        (False, False, True, True, None, 'specclu' ),
        (False, True, True, True, None, 'specclu' ),
        (True, False, True, True, None, 'specclu' ),
        (True, True, True, True, None, 'specclu' ),
        (False, False, False, True, 'cosine', 'specclu' ),
        (False, True, False, True, 'cosine', 'specclu' ),
        (True, False, False, True, 'cosine', 'specclu' ),
        (True, True, False, True, 'cosine', 'specclu' ),
        (False, False, False, True, 'braycurtis', 'specclu' ),
        (False, True, False, True, 'braycurtis', 'specclu' ),
        (True, False, False, True, 'braycurtis', 'specclu' ),
        (True, True, False, True, 'braycurtis', 'specclu' ),
        (False, False, False, True, 'l1', 'specclu' ),
        (False, True, False, True, 'l1', 'specclu' ),
        (True, False, False, True, 'l1', 'specclu' ),
        (True, True, False, True, 'l1', 'specclu' ),
        (False, False, True, True, 'cosine', 'specclu' ),
        (False, True, True, True, 'cosine', 'specclu' ),
        (True, False, True, True, 'cosine', 'specclu' ),
        (True, True, True, True, 'cosine', 'specclu' ),
        (False, False, True, True, 'braycurtis', 'specclu' ),
        (False, True, True, True, 'braycurtis', 'specclu' ),
        (True, False, True, True, 'braycurtis', 'specclu' ),
        (True, True, True, True, 'braycurtis', 'specclu' ),
        (False, False, True, True, 'l1', 'specclu' ),
        (False, True, True, True, 'l1', 'specclu' ),
        (True, False, True, True, 'l1', 'specclu' ),
        (True, True, True, True, 'l1', 'specclu' ),
        (False, False, True, False, None, 'specclu' ),
        (False, True, True, False, None, 'specclu' ),
        (True, False, True, False, None, 'specclu' ),
        (True, True, True, False, None, 'specclu' ),
        (False, False, False, False, 'cosine', 'specclu' ),
        (False, True, False, False, 'cosine', 'specclu' ),
        (True, False, False, False, 'cosine', 'specclu' ),
        (True, True, False, False, 'cosine', 'specclu' ),
        (False, False, False, False, 'braycurtis', 'specclu' ),
        (False, True, False, False, 'braycurtis', 'specclu' ),
        (True, False, False, False, 'braycurtis', 'specclu' ),
        (True, True, False, False, 'braycurtis', 'specclu' ),
        (False, False, False, False, 'l1', 'specclu' ),
        (False, True, False, False, 'l1', 'specclu' ),
        (True, False, False, False, 'l1', 'specclu' ),
        (True, True, False, False, 'l1', 'specclu' ),
        (False, False, True, False, 'cosine', 'specclu' ),
        (False, True, True, False, 'cosine', 'specclu' ),
        (True, False, True, False, 'cosine', 'specclu' ),
        (True, True, True, False, 'cosine', 'specclu' ),
        (False, False, True, False, 'braycurtis', 'specclu' ),
        (False, True, True, False, 'braycurtis', 'specclu' ),
        (True, False, True, False, 'braycurtis', 'specclu' ),
        (True, True, True, False, 'braycurtis', 'specclu' ),
        (False, False, True, False, 'l1', 'specclu' ),
        (False, True, True, False, 'l1', 'specclu' ),
        (True, False, True, False, 'l1', 'specclu' ),
        (True, True, True, False, 'l1', 'specclu' ),

        (False, False, False, False, 'cosine', ('agglo', 'complete') ),
        (False, True, False, False, 'cosine', ('agglo', 'complete') ),
        (True, False, False, False, 'cosine', ('agglo', 'complete') ),
        (True, True,False, False, 'cosine', ('agglo', 'complete') ),
        (False, False, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, False, 'l1', ('agglo', 'single') ),
        (False, True, False, False, 'l1', ('agglo', 'single') ),
        (True, False, False, False, 'l1', ('agglo', 'single') ),
        (True, True, False, False, 'l1', ('agglo', 'single') ),
        (False, False, True, False, 'cosine', ('agglo', 'complete') ),
        (False, True, True, False, 'cosine', ('agglo', 'complete') ),
        (True, False, True, False, 'cosine', ('agglo', 'complete') ),
        (True, True, True, False, 'cosine', ('agglo', 'complete') ),
        (False, False, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, False, 'l1', ('agglo', 'single') ),
        (False, True, True, False, 'l1', ('agglo', 'single') ),
        (True, False, True, False, 'l1', ('agglo', 'single') ),
        (True, True, True, False, 'l1', ('agglo', 'single') ),
        (False, False, False, False, 'cosine', ('agglo', 'complete') ),
        (False, True, False, False, 'cosine', ('agglo', 'complete') ),
        (True, False, False, False, 'cosine', ('agglo', 'complete') ),
        (True, True, False, False, 'cosine', ('agglo', 'complete') ),
        (False, False, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, False, 'l1', ('agglo', 'single') ),
        (False, True, False, False, 'l1', ('agglo', 'single') ),
        (True, False, False, False, 'l1', ('agglo', 'single') ),
        (True, True, False, False, 'l1', ('agglo', 'single') ),
        (False, False, True, False, 'cosine', ('agglo', 'complete') ),
        (False, True, True, False, 'cosine', ('agglo', 'complete') ),
        (True, False, True, False, 'cosine', ('agglo', 'complete') ),
        (True, True, True, False, 'cosine', ('agglo', 'complete') ),
        (False, False, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, False, 'l1', ('agglo', 'single') ),
        (False, True, True, False, 'l1', ('agglo', 'single') ),
        (True, False, True, False, 'l1', ('agglo', 'single') ),
        (True, True, True, False, 'l1', ('agglo', 'single') ),
        (False, False, False, False, 'cosine', ('agglo', 'complete') ),
        (False, True, False, False, 'cosine', ('agglo', 'complete') ),
        (True, False, False, False, 'cosine', ('agglo', 'complete') ),
        (True, True, False, False, 'cosine', ('agglo', 'complete') ),
        (False, False, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, False, 'l1', ('agglo', 'single') ),
        (False, True, False, False, 'l1', ('agglo', 'single') ),
        (True, False, False, False, 'l1', ('agglo', 'single') ),
        (True, True, False, False, 'l1', ('agglo', 'single') ),
        (False, False, True, False, 'cosine', ('agglo', 'complete') ),
        (False, True, True, False, 'cosine', ('agglo', 'complete') ),
        (True, False, True, False, 'cosine', ('agglo', 'complete') ),
        (True, True, True, False, 'cosine', ('agglo', 'complete') ),
        (False, False, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, False, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, False, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, False, 'l1', ('agglo', 'single') ),
        (False, True, True, False, 'l1', ('agglo', 'single') ),
        (True, False, True, False, 'l1', ('agglo', 'single') ),
        (True, True, True, False, 'l1', ('agglo', 'single') ),
        (False, False, False, True, 'cosine', ('agglo', 'complete') ),
        (False, True, False, True, 'cosine', ('agglo', 'complete') ),
        (True, False, False, True, 'cosine', ('agglo', 'complete') ),
        (True, True,False, True, 'cosine', ('agglo', 'complete') ),
        (False, False, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, True, 'l1', ('agglo', 'single') ),
        (False, True, False, True, 'l1', ('agglo', 'single') ),
        (True, False, False, True, 'l1', ('agglo', 'single') ),
        (True, True, False, True, 'l1', ('agglo', 'single') ),
        (False, False, True, True, 'cosine', ('agglo', 'complete') ),
        (False, True, True, True, 'cosine', ('agglo', 'complete') ),
        (True, False, True, True, 'cosine', ('agglo', 'complete') ),
        (True, True, True, True, 'cosine', ('agglo', 'complete') ),
        (False, False, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, True, 'l1', ('agglo', 'single') ),
        (False, True, True, True, 'l1', ('agglo', 'single') ),
        (True, False, True, True, 'l1', ('agglo', 'single') ),
        (True, True, True, True, 'l1', ('agglo', 'single') ),
        (False, False, False, True, 'cosine', ('agglo', 'complete') ),
        (False, True, False, True, 'cosine', ('agglo', 'complete') ),
        (True, False, False, True, 'cosine', ('agglo', 'complete') ),
        (True, True, False, True, 'cosine', ('agglo', 'complete') ),
        (False, False, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, True, 'l1', ('agglo', 'single') ),
        (False, True, False, True, 'l1', ('agglo', 'single') ),
        (True, False, False, True, 'l1', ('agglo', 'single') ),
        (True, True, False, True, 'l1', ('agglo', 'single') ),
        (False, False, True, True, 'cosine', ('agglo', 'complete') ),
        (False, True, True, True, 'cosine', ('agglo', 'complete') ),
        (True, False, True, True, 'cosine', ('agglo', 'complete') ),
        (True, True, True, True, 'cosine', ('agglo', 'complete') ),
        (False, False, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, True, 'l1', ('agglo', 'single') ),
        (False, True, True, True, 'l1', ('agglo', 'single') ),
        (True, False, True, True, 'l1', ('agglo', 'single') ),
        (True, True, True, True, 'l1', ('agglo', 'single') ),
        (False, False, False, True, 'cosine', ('agglo', 'complete') ),
        (False, True, False, True, 'cosine', ('agglo', 'complete') ),
        (True, False, False, True, 'cosine', ('agglo', 'complete') ),
        (True, True, False, True, 'cosine', ('agglo', 'complete') ),
        (False, False, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, False, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, False, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, False, True, 'l1', ('agglo', 'single') ),
        (False, True, False, True, 'l1', ('agglo', 'single') ),
        (True, False, False, True, 'l1', ('agglo', 'single') ),
        (True, True, False, True, 'l1', ('agglo', 'single') ),
        (False, False, True, True, 'cosine', ('agglo', 'complete') ),
        (False, True, True, True, 'cosine', ('agglo', 'complete') ),
        (True, False, True, True, 'cosine', ('agglo', 'complete') ),
        (True, True, True, True, 'cosine', ('agglo', 'complete') ),
        (False, False, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, True, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, False, True, True, 'braycurtis', ('agglo', 'average') ),
        (True, True, True, True, 'braycurtis', ('agglo', 'average') ),
        (False, False, True, True, 'l1', ('agglo', 'single') ),
        (False, True, True, True, 'l1', ('agglo', 'single') ),
        (True, False, True, True, 'l1', ('agglo', 'single') ),
        (True, True, True, True, 'l1', ('agglo', 'single') )


        # (False, False, False, False, 'cosine', 'dbscan' ),
        # (False, True, False, False, 'cosine', 'dbscan' ),
        # (True, False, False, False, 'cosine', 'dbscan' ),
        # (True, True, False, False, 'cosine', 'dbscan' ),
        # (False, False, False, False, 'braycurtis', 'dbscan' ),
        # (False, True, False, False, 'braycurtis', 'dbscan' ),
        # (True, False, False, False, 'braycurtis', 'dbscan' ),
        # (True, True, False, False, 'braycurtis', 'dbscan' ),
        # (False, False, False, False, 'l1', 'dbscan' ),
        # (False, True, False, False, 'l1', 'dbscan' ),
        # (True, False, False, False, 'l1', 'dbscan' ),
        # (True, True, False, False, 'l1', 'dbscan' ),
        # (False, False, True, False, 'cosine', 'dbscan' ),
        # (False, True, True, False, 'cosine', 'dbscan' ),
        # (True, False, True, False, 'cosine', 'dbscan' ),
        # (True, True, True, False, 'cosine', 'dbscan' ),
        # (False, False, True, False, 'braycurtis', 'dbscan' ),
        # (False, True, True, False, 'braycurtis', 'dbscan' ),
        # (True, False, True, False, 'braycurtis', 'dbscan' ),
        # (True, True, True, False, 'braycurtis', 'dbscan' ),
        # (False, False, True, False, 'l1', 'dbscan' ),
        # (False, True, True, False, 'l1', 'dbscan' ),
        # (True, False, True, False, 'l1', 'dbscan' ),
        # (True, True, True, False, 'l1', 'dbscan' ),

        # (False, False, False, True, 'cosine', 'dbscan' ),
        # (False, True, False, True, 'cosine', 'dbscan' ),
        # (True, False, False, True, 'cosine', 'dbscan' ),
        # (True, True, False, True, 'cosine', 'dbscan' ),
        # (False, False, False, True, 'braycurtis', 'dbscan' ),
        # (False, True, False, True, 'braycurtis', 'dbscan' ),
        # (True, False, False, True, 'braycurtis', 'dbscan' ),
        # (True, True, False, True, 'braycurtis', 'dbscan' ),
        # (False, False, False, True, 'l1', 'dbscan' ),
        # (False, True, False, True, 'l1', 'dbscan' ),
        # (True, False, False, True, 'l1', 'dbscan' ),
        # (True, True, False, True, 'l1', 'dbscan' ),
        # (False, False, True, True, 'cosine', 'dbscan' ),
        # (False, True, True, True, 'cosine', 'dbscan' ),
        # (True, False, True, True, 'cosine', 'dbscan' ),
        # (True, True, True, True, 'cosine', 'dbscan' ),
        # (False, False, True, True, 'braycurtis', 'dbscan' ),
        # (False, True, True, True, 'braycurtis', 'dbscan' ),
        # (True, False, True, True, 'braycurtis', 'dbscan' ),
        # (True, True, True, True, 'braycurtis', 'dbscan' ),
        # (False, False, True, True, 'l1', 'dbscan' ),
        # (False, True, True, True, 'l1', 'dbscan' ),
        # (True, False, True, True, 'l1', 'dbscan' ),
        # (True, True, True, True, 'l1', 'dbscan' )
    ]
    return options

if __name__ == '__main__':
    main()

# df = Data().dfs.get('all')
# df = df.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
# smell='ls'

# exclude_outliers=False
# exclude_corr=False
# exclude_spars=False
# n=2
# pca=True
# algorithm='dbscan'
# distance='cosine'
# df = calculate_option(smell, df, smells_df, exclude_outliers, exclude_corr, pca, exclude_spars, n, distance, algorithm)