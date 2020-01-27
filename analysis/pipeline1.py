import pandas as pd
from data import Data
from stats import Stats
from prep import Preprocessing
from significance import Significance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
%config InlineBackend.figure_format='retina'
from utils import scale_df
from utils import flatlist
from utils import allin2
from utils import allin3
from anomaly import AnomalyDetector

data = Data('tosca_and_general', 'all')
data = Stats(data)
a4c_data = Data('tosca_and_general', 'a4c')
a4c_data = Stats(a4c_data)

puc_data = Data('tosca_and_general', 'puccini')
puc_data = Stats(puc_data)

for_data = Data('tosca_and_general', 'forge')
for_data = Stats(for_data)

ex_data = Data('tosca_and_general', 'example')
ex_data = Stats(ex_data)

ind_data = Data('tosca_and_general', 'industry')
ind_data = Stats(ind_data)

top_data = Data('tosca_and_general', 'all', 'topology')
top_data = Stats(top_data)

cus_data = Data('tosca_and_general', 'all', 'custom')
cus_data = Stats(cus_data)

both_data = Data('tosca_and_general', 'all', 'both')
both_data = Stats(both_data)

named_data = {'all' : data, 'a4c' : a4c_data, 'puc' : puc_data, 'for' : for_data, 'ex' : ex_data, 'ind' : ind_data, 'top' : top_data, 'cus' : cus_data, 'both': both_data}





def get_corrs(named_data):
    correlation_df = pd.DataFrame(columns=['features'], index=[list(named_data.keys())])
    for name, dataset in named_data.items():
        corr_features = dataset.corrfeatures
        correlation_df.loc[name, 'features'] = str(corr_features)

    return correlation_df



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

def significance_analysis(datasets):
    '''Not a well created function, only works for a list containing 3 datasets'''

    if len(datasets) == 2:
        allinall = allin2

    if len(datasets) == 3:
        allinall = allin3

    corr_features = [dataset.corrfeatures for dataset in datasets]
    const_features = [dataset.constants for dataset in datasets]
    #all_corr_features = allinall(corr_features)
    any_corr_features = list(set(flatlist(corr_features)))
    all_const_features = allinall(const_features)
    #any_const_features = list(set(flatlist(const_features)))
    print('any correlating features: ', any_corr_features)
    print('all constant features: ', all_const_features)
    print('Number of deleted columns out of 49: ', len(any_corr_features))# + len(all_const_features))
    print('---------------------------------------')

    datasets = [Preprocessing(dataset, corr = any_corr_features) for dataset in datasets]

    #options = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
    options = [[0, 1], [1, 0]]

    if len(options) == 2:
        allinall = allin2

    if len(options) == 3:
        allinall = allin3

    results = []

    for option in options:
        data1 = datasets[option[0]].df
        print('shape before significance: ', data1.shape)
        data2 = datasets[option[1]].df
        #data2 = pd.concat([datasets[option[1]].df, datasets[option[2]].df])
        result = Significance(data1, data2).rejected_features
        results.append(list(result.index))
        print('Option: ', option)
        print('Number of rejected features: ', result.shape)
        print('Rejected features: ', result)
    
    all = allinall(results)
    any = list(set(flatlist(results)))
    twotimes = list(set([x for x in flatlist(results) if flatlist(results).count(x) == 2]))

    print('Rejected in every option: ', len(all))
    print('Rejected in at least one option: ', len(any))
    return twotimes

# file_stats = get_stats(datasets)
# file_stats.to_excel('file_stats.xlsx')


#hier nog ff iets op verzinnen dat je er duidelijk uit krijgt.
#x = get_corrs(named_data)


# Hier ff de pca per verschillende dataset laten zien
# Kan helpen bij het bepalen van evt. clusters 
def pca_insight(dataset):
    original_df = dataset.df

    # Create a PCA instance: pca
    pca = PCA(n_components=5)
    X_std = scale_df(original_df.copy())
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
    plt.show()


# for dataset in datasets:
#     pca_insight(dataset)
#     plt.close


#DROP ANOMALIES 
#%%
def delete_anomalies(datasets, anomalyset):
    ix = AnomalyDetector(anomalyset).outliers
    print('totaal aantal anomalies: ', len(ix))
    filtered_datasets = []
    for dataset in datasets:
        to_drop = [i for i in ix.index if i in dataset.df.index]
        print('originele grootte df: ', dataset.df.shape[0])
        print('aantal anomalies: ', len(to_drop))
        new_df = dataset.df.drop(to_drop)
        print('nieuwe lengte: ', new_df.shape[0])

        filtered_datasets.append(Stats(new_df))

    return filtered_datasets


datasets = [ex_data, ind_data]
significance_analysis(datasets)


# %%

new_dfs = delete_anomalies(datasets, Stats(Data('tosca_and_general', 'all')))
significance_analysis(new_dfs)

# %%
