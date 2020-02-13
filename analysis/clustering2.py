from prep import Preprocessing
from data import Data
import pandas as pd
import numpy as np
import random
from vis import Vis


from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
 
    #Dooie kut return omschrijven
    new_dict = {}
    for key, value in C.items():
        for element in value:
            new_dict[element] = key

    new_list = [None] * len(new_dict)
    for key, value in new_dict.items():
        new_list[key] = value

    labels = np.array(new_list)
    # return results
    return M, labels


def clustering(array, algo, k):
    '''For k-Medoids an esemble is implemented of 100 iterations'''
#Heeft dit wel zin? Want wat je nu ziet is dat ie elke keer een andere (random?) cluster pakt terwijl ze misschien
#wel hetzelfde zijn, maar qua naam anders zijn. Dit is echter voor alle punten, dus als het goed is komen deze 
#als nog bij elkaar te zitten

    if algo == 'kmedoids':
        label_dict = {i : [] for i in range(len(array))}
        i=0
        while i < 200:
            m, labels = kMedoids(array, k)

            for key, value in label_dict.items():
                value.append(labels[key])
            i += 1
   
        labels = [None] * len(array)
        for key, value in label_dict.items():
            value = np.array(value)
            counts = np.bincount(value)
            labels[key] = np.argmax(counts)
        print('Iterations: ', i)

        return np.array(labels)

    elif algo == 'agglo':
        model = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
        return model.fit_predict(array)
    elif algo == 'gm':
        model = GaussianMixture(n_components=k)
        return model.fit_predict(array)

    else:
        print('Invalid algo, only "kmedoids", "agglo" and "gm"')

def calculate_cluster_score(df, k_sizes, algos, metrics):
    index = pd.MultiIndex.from_product(iterables=[algos.keys(), metrics.keys()], names=['algo', 'evaluation'])
    scores = pd.DataFrame(index=index, columns=k_sizes)

    for k in k_sizes:
        for algo_name, algo in algos.items():
            result = clustering(df, algo, k)
            for metric_name, metric in metrics.items():
                score = metric(df, result)
                scores.loc[(algo_name, metric_name), k] = score
    return scores


k_sizes = [2,3,4,5,6,7,8]

algos = {'kmedoids' : 'kmedoids', \
    'agglo' : 'agglo', \
    'gm' : 'gm'}


metrics = {'silhouette' : silhouette_score, \
    'davies' : davies_bouldin_score, 
    'calinski' : calinski_harabasz_score}

data = Data('tosca_and_general', 'all')
dist_dict = {#'spearman' : Preprocessing(data, customdistance='spearman'),
    'braycurtis' : Preprocessing(data, customdistance='braycurtis'),
    'cosine' : Preprocessing(data, customdistance='cosine'),
    'l1' : Preprocessing(data, customdistance='l1')}

#%% 
results = {}
evals = {}

# #TIJDELIJK
# data = Data('tosca_and_general', 'all')
# dist_dict = {'cosine' : Preprocessing(data, customdistance='cosine')}

for key, data in dist_dict.items():
    # #tijdelijk
    # from sklearn.metrics import pairwise_distances
    # data = Data('tosca_and_general', 'all').df.to_numpy()
    # data = pairwise_distances(data, metric='cosine')

    try:
        resu = clustering(data.df.to_numpy(), 'gm', 4)
        #resu = clustering(data, 'agglo', 4)
        unique, counts = np.unique(resu, return_counts=True)
        results[key] = resu #dict(zip(unique, counts))
    except Exception:
        results[key] = 'all in one cluster so error'
        
    print(key, ':', np.unique(results[key], return_counts=True)    )
    # try:
    #     eva = calculate_cluster_score(data.df.to_numpy(), k_sizes, algos, metrics)
    #     #eva = calculate_cluster_score(data, k_sizes, algos, metrics)
    #     evals[key] = eva
    # except Exception:
    #     evals[key] = 'all in one cluster so error'

#%%
#HIER DE STATS PER CLUSTER EVEN BEKIJKEN, CELL HIERBOVEN IS NODIG!!
from stats import Stats
import pickle

ori_df = Data('tosca_and_general', 'all').df
distance = 'braycurtis'
ori_df['cluster'] = results[distance]
pickle.dump(ori_df, open('../temp_data/dfpluscluster_{}'.format(distance), 'wb'))

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

#NOG SPLITSEN OP CLUSTER!! DONE
clu0 = ori_df[ori_df['cluster'] == 0]
clu1 = ori_df[ori_df['cluster'] == 1]
clu2 = ori_df[ori_df['cluster'] == 2]
clu3 = ori_df[ori_df['cluster'] == 3]

datasets = [Stats(clu0), Stats(clu1), Stats(clu2), Stats(clu3)]
stat_results = get_stats(datasets)
vis = Vis(ori_df, 'gm2braycurtis')

#%%#########################################################################
##                           feature importance                           ##
############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def feature_extraction(df, predictor):
    copy_df = df.copy()
    y = copy_df['cluster']
    X = copy_df.drop(columns=['cluster'])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    predictor.fit(X=X_train, y=y_train)
    score = predictor.score(X_test, y_test)
    imps = predictor.feature_importances_
    feature_importances = sorted(zip(X_test, imps), reverse=True, key=lambda x: x[1])
    feature_importances.append(('pred_score', score))        
    return feature_importances

feature_extraction(ori_df, RandomForestClassifier())


#%%#########################################################################
##                 PCA analysis and cluster visualisation                 ##
############################################################################
#from  https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
# and  https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import scale_df

# Create a PCA instance: pca
pca = PCA(n_components=5)
X_std = scale_df(Data('tosca_and_general', 'all').df.copy())
X_std.values
principalComponents = pca.fit_transform(X_std)

# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

# Check visually for the existance of clusters
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

#Visualize 

print(ori_df['cluster'].value_counts())
df = ori_df.reset_index(drop=False)
finalDf = pd.concat([PCA_components, df[['cluster']]], axis = 1)

#Plot the 2 component PCA with the found clusters by the algorithm
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)

clusters = [0, 1, 2, 3, 4]
colors = ['r', 'g', 'b', 'y', 'c']
for cluster, color in zip(clusters,colors):
    indicesToKeep = finalDf['cluster'] == cluster
    ax.scatter(finalDf.loc[indicesToKeep, 0]
            , finalDf.loc[indicesToKeep, 1]
            , c = color
            , s = 50)
ax.legend(clusters)
ax.grid()






#%%

#HIER ALLE VERGELIJKINGEN MET DE ORIGINELE DATASETS GEDAAN OM TE KIJKEN OF ZE DAARMEE OVERLAP HEBBEN
#NIET HET GEVAL


def compare_indexes_repo():
    # data = Data('tosca_and_general', 'all')
    # data = Preprocessing(data, customdistance='braycurtis')

    # a4c_data = Data('tosca_and_general', 'a4c').df
    # puc_data = Data('tosca_and_general', 'puccini').df
    # forg_data = Data('tosca_and_general', 'forge').df

    from sklearn.metrics import pairwise_distances
    from utils import scale_df

    a4c_data = Data('tosca_and_general', 'a4c').df
    puc_data = Data('tosca_and_general', 'puccini').df
    forg_data = Data('tosca_and_general', 'forge').df
    all_data = a4c_data.append(puc_data)
    all_data = all_data.append(forg_data) 

    df = all_data.copy()
    scaled = scale_df(df)

    distances = pairwise_distances(scaled, metric='braycurtis')
    matrix = pd.DataFrame(data=distances, index=all_data.index, columns=all_data.index)




    resu = clustering(matrix.to_numpy(), 'gm', 4)
    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 0], :]

    print('out of', len(df), 'from the first cluster: ')
    

    count = {'a4c' : 0, 'forge' : 0, 'puccini' : 0}
    a4c = df.index.str.contains('A4C')
    count['a4c'] = np.sum(a4c)
    forge = df.index.str.contains('Forge')
    count['forge'] = np.sum(forge)
    puc = df.index.str.contains('Puccini')
    count['puccini'] = np.sum(puc)

    print(count)
    print('\n')

    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 1], :]

    print('out of', len(df), 'from the second cluster: ')

    count = {'a4c' : 0, 'forge' : 0, 'puccini' : 0}
    a4c = df.index.str.contains('A4C')
    count['a4c'] = np.sum(a4c)
    forge = df.index.str.contains('Forge')
    count['forge'] = np.sum(forge)
    puc = df.index.str.contains('Puccini')
    count['puccini'] = np.sum(puc)

    print(count)
    print('\n')

    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 2], :]

    print('out of', len(df), 'from the third cluster: ')

    count = {'a4c' : 0, 'forge' : 0, 'puccini' : 0}
    a4c = df.index.str.contains('A4C')
    count['a4c'] = np.sum(a4c)
    forge = df.index.str.contains('Forge')
    count['forge'] = np.sum(forge)
    puc = df.index.str.contains('Puccini')
    count['puccini'] = np.sum(puc)

    print(count)
    print('\n')

    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 3], :]

    print('out of', len(df), 'from the fourth cluster: ')

    count = {'a4c' : 0, 'forge' : 0, 'puccini' : 0}
    a4c = df.index.str.contains('A4C')
    count['a4c'] = np.sum(a4c)
    forge = df.index.str.contains('Forge')
    count['forge'] = np.sum(forge)
    puc = df.index.str.contains('Puccini')
    count['puccini'] = np.sum(puc)

    print(count)
    print('\n')
    
    return

compare_indexes_repo()



# %%
def compare_indexes_purpose():
    all_data = Data('tosca_and_general', 'all').df
    # data = Preprocessing(data, customdistance='braycurtis')

    # a4c_data = Data('tosca_and_general', 'a4c').df
    # puc_data = Data('tosca_and_general', 'puccini').df
    # forg_data = Data('tosca_and_general', 'forge').df

    from sklearn.metrics import pairwise_distances
    from utils import scale_df

    df = all_data.copy()
    scaled = scale_df(df)

    distances = pairwise_distances(scaled, metric='braycurtis')
    matrix = pd.DataFrame(data=distances, index=all_data.index, columns=all_data.index)




    resu = clustering(matrix.to_numpy(), 'gm', 4)
    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 0], :]

    print('out of', len(df), 'from the first cluster: ')
    

    count = {'industry' : 0, 'example' : 0}
    industry = df.index.str.contains('Total Industry')
    count['industry'] = np.sum(industry)
    example = df.index.str.contains('Total Examples')
    count['example'] = np.sum(example)

    print(count)
    print('\n')

    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 1], :]

    print('out of', len(df), 'from the second cluster: ')

    count = {'industry' : 0, 'example' : 0}
    industry = df.index.str.contains('Total Industry')
    count['industry'] = np.sum(industry)
    example = df.index.str.contains('Total Examples')
    count['example'] = np.sum(example)

    print(count)
    print('\n')

    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 2], :]

    print('out of', len(df), 'from the third cluster: ')

    count = {'industry' : 0, 'example' : 0}
    industry = df.index.str.contains('Total Industry')
    count['industry'] = np.sum(industry)
    example = df.index.str.contains('Total Examples')
    count['example'] = np.sum(example)

    print(count)
    print('\n')

    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 3], :]

    print('out of', len(df), 'from the fourth cluster: ')

    count = {'industry' : 0, 'example' : 0}
    industry = df.index.str.contains('Total Industry')
    count['industry'] = np.sum(industry)
    example = df.index.str.contains('Total Examples')
    count['example'] = np.sum(example)

    print(count)
    print('\n')

compare_indexes_purpose()

# %%

def compare_indexes_type():
    all_data = Data('tosca_and_general', 'all').df
    top_data = Data('tosca_and_general', 'all', 'topology').df
    cus_data = Data('tosca_and_general', 'all', 'custom').df
    both_data = Data('tosca_and_general', 'all', 'both').df

    from sklearn.metrics import pairwise_distances
    from utils import scale_df

    df = all_data.copy()
    scaled = scale_df(df)

    distances = pairwise_distances(scaled, metric='braycurtis')
    matrix = pd.DataFrame(data=distances, index=all_data.index, columns=all_data.index)


    resu = clustering(matrix.to_numpy(), 'gm', 4)
    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 0], :]

    print('out of', len(df), 'from the first cluster: ')

    top = 0
    cus = 0
    both = 0

    for ix in df.index:
        if ix in top_data.index:
            top += 1

        if ix in cus_data.index:
            cus += 1

        if ix in both_data.index:
            both += 1
    
    print({'top' : top, 'cus' : cus, 'both' : both})
    print('\n')

    
    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 1], :]

    print('out of', len(df), 'from the second cluster: ')

    top = 0
    cus = 0
    both = 0

    for ix in df.index:
        if ix in top_data.index:
            top += 1

        if ix in cus_data.index:
            cus += 1

        if ix in both_data.index:
            both += 1
    
    print({'top' : top, 'cus' : cus, 'both' : both})
    print('\n')

    
    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 2], :]

    print('out of', len(df), 'from the third cluster: ')

    top = 0
    cus = 0
    both = 0

    for ix in df.index:
        if ix in top_data.index:
            top += 1

        if ix in cus_data.index:
            cus += 1

        if ix in both_data.index:
            both += 1

    print({'top' : top, 'cus' : cus, 'both' : both})
    print('\n')

    df = all_data.iloc[[i for i, e in enumerate(resu) if e == 3], :]

    print('out of', len(df), 'from the fourth cluster: ')

    top = 0
    cus = 0
    both = 0

    for ix in df.index:
        if ix in top_data.index:
            top += 1

        if ix in cus_data.index:
            cus += 1

        if ix in both_data.index:
            both += 1
    
    print({'top' : top, 'cus' : cus, 'both' : both})
    print('\n')

    

compare_indexes_type()

# %%


# %%
