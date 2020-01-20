from data import Data

datasets = ['all' ,'industry' , 'example' ,'a4c', 'forge', 'puccini']
metric_types = ['general', 'tosca', 'tosca_and_general']
             
for ds in datasets:
    for met in metric_types:
        Data(met, ds)