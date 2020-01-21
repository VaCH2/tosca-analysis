import pandas as pd
from data import Data
from stats import Stats
from prep import Preprocessing
from significance import Significance

data = Data('tosca_and_general', 'all')
a4c_data = Data('tosca_and_general', 'a4c')
puc_data = Data('tosca_and_general', 'puccini')
for_data = Data('tosca_and_general', 'forge')

#significance test for each with the total set: is there a difference?
#Nee niet met total want dat kan dezelfde file erin zitten!!
datasets = [a4c_data, puc_data, for_data]
options = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]

results = []

for option in options:
    data1 = datasets[option[0]].df
    data2 = pd.concat([datasets[option[1]].df, datasets[option[2]].df])
    result = Significance(data1, data2).rejected_features
    results.append(list(result.index))
    print(result.index)

#nog even bedenken hoe ik dit fatoenlijk wil returnen
#functie welke features voor alle combinaties rejected zijn. 
#zijn er 9 van de 49, wat zegt dit?

all = set()

for element in [item for sublist in results for item in sublist]:
    if element in results[0]:
        if element in results[1]:
            if element in results[2]:
                all.add(element)

# data_stats = Stats(data)
# prep = Preprocessing(data_stats, constants=False, corr=False)

