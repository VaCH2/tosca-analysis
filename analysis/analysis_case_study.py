import itertools
import os
import pickle
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from clusterevaluator import SmellEvaluator
from clusterconfigurator import ClusterConfigurator
from data import Data 
import cliffsDelta

#Rule-based Detector
from rulebased.detector import toomanyattributes
from rulebased.detector import duplicateblocks
from rulebased.detector import insufficientmodularization

from imblearn.over_sampling import RandomOverSampler 
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from scipy.stats import mannwhitneyu

root_folder = os.path.dirname(os.path.dirname( __file__ ))
results_folder = os.path.join(root_folder, 'results', 'case_study')
data_folder = os.path.join(root_folder, 'dataminer', 'tmp')
temp_folder = os.path.join(root_folder, 'temp_data', 'case_study')

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)


#--------------Table
db = SmellEvaluator('db')
tma = SmellEvaluator('tma')
im = SmellEvaluator('im')

# # #--------------Internal and External measurement figure

# def resetIndex(df):
#     df['newIndex'] = [i for i in range(1, (df.shape[0] + 1))]
#     df = df.set_index(keys='newIndex', drop=True)
#     return df

# dfsInternal = {
#     'db' : resetIndex(db.evalDf).head(5),
#     'tma' : resetIndex(tma.evalDf).head(5),
#     'im' : resetIndex(im.evalDf).head(5) 
# }

# colors = {
#     'db' : '#0057e7',
#     'tma' : '#d62d20',
#     'im' : '#ffa700'
# }

# names = {
#     'db' : 'Duplicate Block',
#     'tma' : 'Too many Attributes',
#     'im' : 'Insufficient Modularization'
# }

# def createTrace(smell, pm, legend):
#     df = dfsInternal[smell]
#     trace = dict(
#         type = 'scatter',
#         x = df.index,
#         y = df[pm],
#         mode = 'lines',
#         line = dict(color = colors[smell], width=6),
#         name = names[smell],
#         showlegend = legend
#     )
#     return trace

# fig = make_subplots(rows=6, cols=1, shared_xaxes=True, subplot_titles=['Silhouette Score (Higher is better)', 'Calinski-Harabasz Index (Higher is better)', 'Davies-Bouldin  Index (Lower is better)', 'Precision (Higher is better)', 'Matthews Correlation Coefficient (Higher is better)', 'Adjusted Rand Index (Higher is better)'], vertical_spacing = 0.05)

# legend = True
# for ix, pm in enumerate(['sc', 'ch', 'db', 'precision', 'mcc', 'ari']):
#     ix += 1
#     fig.append_trace(createTrace('db', pm, legend), ix, 1)
#     fig.append_trace(createTrace('tma', pm, legend), ix, 1)
#     fig.append_trace(createTrace('im', pm, legend), ix, 1)
#     legend = False

# fig.update_layout(
#     height=2900, 
#     width=2800, 
#     title_text="Top 5 Configurations",
#     paper_bgcolor='rgba(255, 255, 255, 1)',
#     plot_bgcolor='rgba(255, 255, 255, 1)',
#     legend=dict(x=-.1, y=1.2, orientation='h'),
#     font = dict(size=47)
#     )

# #fix subplot title size
# for i in fig['layout']['annotations']:
#     i['font'] = dict(size=47)

# fig.show()
# #fig.write_image(os.path.join(results_folder, 'configurationperformance.png'))


# #-----------Stability
# dbTopConfigs = [
#     (False, False, False, True, None, ('gm', 'spherical')),
#     (False, False, True, True, None, ('gm', 'spherical')),
#     (False, False, True, True, 'braycurtis', ('kmedoids', None)),
#     (False, False, False, True, None, ('gm', 'tied')),
#     (False, True, True, True, None, ('gm', 'full')),
# ]
 
# tmaTopConfigs = [
#     (False, False, False, True, None, ('gm', 'full')),
#     (False, False, False, True, None, ('gm', 'tied')),
#     (False, False, False, True, None, ('gm', 'spherical')),
#     (False, False, True, True, None, ('gm', 'spherical')),
#     (False, False, False, True, 'l1', ('agglo', 'complete'))
# ]


# imTopConfigs = [
#     (False, False, False, True, None, ('gm', 'full')),
#     (False, False, False, True, None, ('gm', 'tied')),
#     (False, False, True, True, None, ('gm', 'full')),
#     (False, False, True, True, None, ('gm', 'spherical')),
#     (False, False, False, True, None, ('gm', 'spherical'))
# ]

# for ix, dbTopConfig in enumerate(dbTopConfigs):
#     tempDf = db.df.copy(deep=True)
#     tempDf = tempDf.drop(['index'], axis=1)
#     dbTopConfigModel = ClusterConfigurator(tempDf, dbTopConfig)
#     stability = dbTopConfigModel.getStability()
#     print(f'DB config {ix}: {stability[0]}')

# for ix, tmaTopConfig in enumerate(tmaTopConfigs):
#     tempDf = tma.df.copy(deep=True)
#     tempDf = tempDf.drop(['index'], axis=1)
#     tmaTopConfigModel = ClusterConfigurator(tempDf, tmaTopConfig)
#     stability = tmaTopConfigModel.getStability()
#     print(f'TmA config {ix}: {stability[0]}')

# for ix, imTopConfig in enumerate(imTopConfigs):
#     tempDf = im.df.copy(deep=True)
#     tempDf = tempDf.drop(['index'], axis=1)
#     imTopConfigModel = ClusterConfigurator(tempDf, imTopConfig)
#     stability = imTopConfigModel.getStability()
#     print(f'IM config {ix}: {stability[0]}')


# #----------------------Comparision
# #Here, we calculate the MCC and Precision for 100 subsamples
# #We create boxplots and a statistical test to compare the two detectors

# subSample = 70
# iters = 100

# def comparison():
#     blueprintLabels = getGroundTruth()
#     scoreDict = {
#         'rule-db-mcc' : [],
#         'rule-db-precision' : [],
#         'rule-tma-mcc' : [],
#         'rule-tma-precision' : [],
#         'rule-im-mcc' : [],
#         'rule-im-precision' : [],
#         'cluster-db-mcc' : [],
#         'cluster-db-precision' : [],
#         'cluster-tma-mcc' : [],
#         'cluster-tma-precision' : [],
#         'cluster-im-mcc' : [],
#         'cluster-im-precision' : [],
#     }

#     try:
#         scoreDict = pickle.load(open(os.path.join(temp_folder, f'MCCandPrecisionFor{iters}iters{subSample}percent'), 'rb'))

#     except (OSError, IOError):
#         for i in range(iters):
#             print('Iteration: ', i)
#             sampleLabels = blueprintLabels.sample(frac=subSample/100)

#             for smell in ['db', 'tma', 'im']:
#                 ruleLabels = getRuleLabels(sampleLabels.index, smell)
#                 clusterLabels = getClusterLabels(sampleLabels.index, smell)
                
#                 #Ensure same cluster shape when outliers are dropped
#                 sampleLabels = sampleLabels[sampleLabels.index.isin(clusterLabels.index)]
#                 sampleLabels = sampleLabels.sort_index()
#                 ruleLabels = ruleLabels[ruleLabels.index.isin(clusterLabels.index)]

#                 scoreDict[f'rule-{smell}-mcc'].append(calculateScore('mcc', sampleLabels[smell], ruleLabels))
#                 scoreDict[f'rule-{smell}-precision'].append(calculateScore('precision', sampleLabels[smell], ruleLabels))
#                 scoreDict[f'cluster-{smell}-mcc'].append(calculateScore('mcc', sampleLabels[smell], clusterLabels))
#                 scoreDict[f'cluster-{smell}-precision'].append(calculateScore('precision', sampleLabels[smell], clusterLabels))
    
#         pickle.dump(scoreDict, open(os.path.join(temp_folder, f'MCCandPrecisionFor{iters}iters{subSample}percent'), 'wb'))  
    

#     statisticalTest(scoreDict)
#     boxplotCreation(scoreDict)

# def getGroundTruth():
#     smells = pd.read_excel('results/labeling/to_label.xlsx', sheet_name='Sheet1', usecols='B,E,D,G', nrows=685, index_col=0)
#     smells = smells.drop(r'SeaCloudsEU\tosca-parser\Industry\Noart.tomcat-DC-compute-mysql-compute.yaml')
#     return smells.astype(bool)

# def calculteRule(path, smell):
#     if smell is 'db':
#         label = duplicateblocks.evaluate_script_with_rule(path)
#     elif smell is 'tma':
#         label = toomanyattributes.evaluate_script_with_rule(path)
#     elif smell is 'im':
#         label = insufficientmodularization.evaluate_script_with_rule(path)
#     return label


# def getRuleLabels(ix, smell):
#     try:
#         ruleDf = pickle.load(open(os.path.join(temp_folder, 'rulebasedlabels'), 'rb'))

#     except (OSError, IOError):

#         results = {}

#         for blueprint in getGroundTruth().index:
#             path = os.path.join(data_folder, blueprint)
#             results[blueprint] = {
#                 'tma' : toomanyattributes.evaluate_script_with_rule(path),
#                 'db' : duplicateblocks.evaluate_script_with_rule(path),
#                 'im' : insufficientmodularization.evaluate_script_with_rule(path),
#             }

#         ruleDf = pd.DataFrame(results).T
#         ruleDf = ruleDf.astype(bool)  
#         pickle.dump(ruleDf, open(os.path.join(temp_folder, 'rulebasedlabels'), 'wb'))

#     ruleDf = ruleDf[ruleDf.index.isin(ix)].sort_index()
#     return ruleDf[smell]


# def constructDf(ix, smell):
#     '''Copied from the clusterEvaluator class to enable data balancing.
#     Afterwards, we filterout the identified indexes of the subset again'''
#     if smell == 'db':
#         clusterDf = db.df
#     elif smell == 'tma':
#         clusterDf = tma.df
#     elif smell == 'im':
#         clusterDf = im.df
#     clusterDf = clusterDf.set_index('index')
#     clusterDf = clusterDf[clusterDf.index.isin(ix)]
#     clusterDf = clusterDf.loc[~clusterDf.index.duplicated(keep='first')]
#     return clusterDf.sort_index()

# def getClusterLabels(ix, smell):
#     df = constructDf(ix, smell)

#     bestConfigurations = {
#         'db' : (False, False, False, True, None, ('gm', 'spherical')),
#         'tma' : (False, False, True, True, None, ('gm', 'spherical')),
#         'im' : (False, False, True, True, None, ('gm', 'spherical'))
#     }

#     configInstance = ClusterConfigurator(df, bestConfigurations[smell])
#     return configInstance.labels['cluster']

# def calculateScore(pm, trueLabels, predLabels):
#     trueLabels = trueLabels.map({True: 1, False: 0})
#     predLabels = predLabels.map({True: 1, False: 0})

#     if pm is 'mcc':
#         score = matthews_corrcoef(trueLabels, predLabels)
#     elif pm is 'precision':
#         score = precision_score(trueLabels, predLabels)
    
#     return score

# def statisticalTest(scoreDict):
#     pairs = {
#         'db-mcc' : ('rule-db-mcc', 'cluster-db-mcc'),
#         'db-precision' : ('rule-db-precision', 'cluster-db-precision'),        
#         'tma-mcc' : ('rule-tma-mcc', 'cluster-tma-mcc'),
#         'tma-precision' : ('rule-tma-precision', 'cluster-tma-precision'),     
#         'im-mcc' : ('rule-im-mcc', 'cluster-im-mcc'),
#         'im-precision' : ('rule-im-precision', 'cluster-im-precision'),     
#     }
#     uDict = {}
#     effectDict = {}

#     for name, pair in pairs.items():
#         stat, p = mannwhitneyu(np.array(scoreDict[pair[0]]), np.array(scoreDict[pair[1]]))
#         uDict[name] = (stat, p)
#         effectsize, res = cliffsDelta.cliffsDelta(scoreDict[pair[0]], scoreDict[pair[1]])
#         effectDict[name] = (effectsize, res)
    
#     uDf = pd.DataFrame(data=uDict).T.to_excel(os.path.join(results_folder, f'utestresults{iters}iters{subSample}percent.xlsx'))
#     effectDf = pd.DataFrame(data=effectDict).T.to_excel(os.path.join(results_folder, f'effectresults{iters}iters{subSample}percent.xlsx'))        

# def boxplotCreation(scoreDict):
#     def createTrace(combi, scoreDict):
#         scoreList = scoreDict[combi]
#         #Hier gaat nog iets niet goed
#         detector, smell, pm = combi.split('-')
#         if detector == 'rule':
#             detector = 'Rule-Based'
#         else:
#             detector = 'Cluster'
        
#         colors = {
#             'db' : '#0057e7',
#             'tma' : '#d62d20',
#             'im' : '#ffa700'
#         }

#         box = go.Box(
#             y=scoreList,
#             name=detector,
#             boxpoints='all',
#             marker_color=colors[smell],
#             line_color=colors[smell]
#         )
#         return box
    
#     #MCC
#     fig = make_subplots(rows=1, cols=6, shared_yaxes=True, subplot_titles=['Duplicate Block', '', 'Too many Attributes', '', 'Insufficient Modularization', ''])
#     for ix, combi in enumerate(['rule-db-mcc', 'cluster-db-mcc', 'rule-tma-mcc', 'cluster-tma-mcc', 'rule-im-mcc', 'cluster-im-mcc']):
#         ix += 1
#         fig.append_trace(createTrace(combi, scoreDict), 1, ix)
#     fig.update_layout(
#         height=1300, 
#         width=2800, 
#         paper_bgcolor='rgba(255, 255, 255, 1)',
#         plot_bgcolor='rgba(255, 255, 255, 1)',
#         showlegend=False,
#         font = dict(size=47)
#         )
#     #fix subplot title size
#     for i in fig['layout']['annotations']:
#        i['font'] = dict(size=47)
#     fig.show()
    
#     #fig.write_image(os.path.join(results_folder, f'comparison50sampleMCC{iters}iters{subSample}percent.png'))


#     #Precision
#     fig = make_subplots(rows=1, cols=6, shared_yaxes=True, subplot_titles=['Duplicate Block', '', 'Too many Attributes', '', 'Insufficient Modularization', ''])
#     for ix, combi in enumerate(['rule-db-precision', 'cluster-db-precision', 'rule-tma-precision', 'cluster-tma-precision', 'rule-im-precision', 'cluster-im-precision']):
#         ix += 1
#         fig.append_trace(createTrace(combi, scoreDict), 1, ix)
#     fig.update_layout(
#         height=1300, 
#         width=2800, 
#         paper_bgcolor='rgba(255, 255, 255, 1)',
#         plot_bgcolor='rgba(255, 255, 255, 1)',
#         showlegend=False,
#         font = dict(size=47),
#         yaxis=dict(range=[0,0.7])
#         )
#     #fix subplot title size
#     for i in fig['layout']['annotations']:
#        i['font'] = dict(size=47)
#     fig.show()
    
#     #fig.write_image(os.path.join(results_folder, f'comparison50sampleprecision{iters}iters{subSample}percent.png'))

# comparison()