#%%

import pandas as pd
from data import Data
from stats import Stats
from prep import Preprocessing
from significance import Significance

data = Data('tosca_and_general', 'all')
data = Stats(data)
a4c_data = Data('tosca_and_general', 'a4c')
a4c_data = Stats(a4c_data)

puc_data = Data('tosca_and_general', 'puccini')
puc_data = Stats(puc_data)

for_data = Data('tosca_and_general', 'forge')
for_data = Stats(for_data)

datasets = [a4c_data, puc_data, for_data]

def flatlist(nested_list):
    return [item for sublist in nested_list for item in sublist]

def allinall(nested_list):
    all = set()

    for element in flatlist(nested_list):
        if element in nested_list[0]:
            if element in nested_list[1]:
                if element in nested_list[2]:
                    all.add(element)
    return list(all)

#significance test for each with the total set: is there a difference?
#Nee niet met total want dat kan dezelfde file erin zitten!!
#nog even bedenken hoe ik dit fatoenlijk wil returnen
#functie welke features voor alle combinaties rejected zijn. 
#zijn er 9 van de 49, wat zegt dit?
#Dit dadelijk doen voor de andere mogelijkheden
#Combineren met statistics om iets over te zeggen?

def significance_analysis(datasets):
    '''Not a well created function, only works for a list containing 3 datasets'''

    corr_features = [dataset.corrfeatures for dataset in datasets]
    #all_corr_features = allinall(corr_features)
    any_corr_features = list(set(flatlist(corr_features)))
    print('Number of deleted columns out of 49: ', len(any_corr_features))

    datasets = [Preprocessing(dataset, corr = any_corr_features) for dataset in datasets]

    options = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]

    results = []

    for option in options:
        data1 = datasets[option[0]].df
        data2 = pd.concat([datasets[option[1]].df, datasets[option[2]].df])
        result = Significance(data1, data2).rejected_features
        results.append(list(result.index))
    
    all = allinall(results)
    any = list(set(flatlist(results)))
    print('Rejected in every option: ', len(all))
    print('Rejected in at least one option: ', len(any))
    return all

significance_analysis(datasets)
