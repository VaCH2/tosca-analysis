
import itertools
import os
import pickle
import pandas as pd
from clusterconfigurator import ClusterConfigurator
from data import Data
from imblearn.over_sampling import RandomOverSampler 


class SmellEvaluator():

    root_folder = os.path.dirname(os.path.dirname( __file__ ))
    results_folder = os.path.join(root_folder, 'results', 'clustering_models')

    def __init__(self, smell):        
        self.smell = smell
        self.df = self.constructDf(self.smell)
        self.configs = self.getConfigurations()
        self.evalDf = self.configCalculationAndEvaluation(self.df, self.configs)
        self.topconfig = self.getTopConfig(self.evalDf)

    
    def getData(self):
        data = Data().dfs.get('all')
        data = data.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
        return data

    def getSmells(self):
        smells = pd.read_excel('../results/labeling/to_label.xlsx', sheet_name='Sheet1', usecols='B:H', nrows=685, index_col=0)
        return smells.astype(bool)

    def oversampleData(self, df):
        oversample = RandomOverSampler(sampling_strategy=0.5, random_state=1)
        oversampleDf, _ = oversample.fit_resample(df, df['smell'])
        return oversampleDf

    def constructDf(self, smell):
        smellSeries = self.getSmells()[smell].rename('smell')
        df = self.getData()
        df = df.merge(smellSeries, how='inner', left_index=True, right_index=True)
        df = self.oversampleData(df)
        return df

    def getConfigurations(self):
        ex_out = ex_cor = pca = ex_spr = [True, False]
        prep = [ex_out, ex_cor, pca, ex_spr]
        algos = [
            ('agglo', 'complete'), ('agglo', 'average'), ('agglo', 'single'), 
            ('kmedoids', None), ('specclu', None), 
            ('gm', 'full'), ('gm', 'tied'), ('gm', 'diag'), ('gm', 'spherical')
            ]
        distance = ['braycurtis', 'cosine', 'l1', None]

        configurations = []
        for algo in algos:
            if algo[0] is 'gm':
                distance = [None]
            elif algo[0] is 'specclu':
                distance = ['braycurtis', 'cosine', 'l1', None]
            else:
                distance = ['braycurtis', 'cosine', 'l1']

            distance_perm = list(itertools.product(*[distance, [algo]]))
            prep_perm = list(itertools.product(*prep))
            total_perm = list(itertools.product(*[prep_perm, distance_perm]))
            total_perm = [(t[0][0], t[0][1], t[0][2], t[0][3], t[1][0], t[1][1]) for t in total_perm]
            
            configurations.extend(total_perm)
        return configurations

    def c2s(self, smell, config):
        '''Config to string'''
        return f'smell={smell}_exout={config[0]}_excor={config[1]}_pca={config[2]}_exspr={config[3]}_{config[4]}_{config[5][0]}_{config[5][1]}'

    def getPickle(self, smell, config):
        return pickle.load(open(os.path.join(self.root_folder, 'temp_data', 'clustering', self.c2s(smell, config)), 'rb'))

    def setPickle(self, smell, config, instance):
        pickle.dump(instance, open(os.path.join(self.root_folder, 'temp_data', 'clustering', self.c2s(smell, config)), 'wb'))


    def configCalculationAndEvaluation(self, df, configs):
        scoreDict = {}

        for config in configs:
            try:
                configInstance = self.getPickle(self.smell, config)
            except (OSError, IOError):
                configInstance = ClusterConfigurator(df, config)
                self.setPickle(self.smell, config, configInstance)
            
            scores = configInstance.scores
            distribution = configInstance.labels['cluster'].value_counts()
            scores['smellySize'] = distribution.loc[True]
            scores['soundSize'] = distribution.loc[False]
            scoreDict[self.c2s(self.smell, config)] = scores

        scoreDf = pd.DataFrame.from_dict(scoreDict, orient='index', columns=['sc', 'ch', 'db', 'precision', 'mcc', 'ari', 'soundSize', 'smellySize'])
        evalDf = self.scoreAggregation(scoreDf, config)
        evalDf['total_score_percentage'] = (evalDf['total_score'] / 1920) * 100
        return evalDf


    def scoreAggregation(self, scoreDf, config):
        evalDf = scoreDf.copy(deep=True)
        evalDf['total_score'] = 0
        evalDf = evalDf.reset_index()

        for pm in scoreDf.columns:
            if pm is 'db':
                evalDf = evalDf.sort_values(by=pm, ascending=True)
            elif pm in ['smellySize', 'soundSize']:
                break
            else:
                evalDf = evalDf.sort_values(by=pm, ascending=False)
            evalDf = evalDf.reset_index(drop=True)
            evalDf['total_score'] = evalDf['total_score'] + evalDf.shape[0] - evalDf.index.values

        evalDf = evalDf.set_index('index')
        evalDf = evalDf.sort_values(by='total_score', ascending=False)
        return evalDf

    def getTopConfig(self, evalDf):
        return evalDf.iloc[0]




#------------- case study
import plotly.graph_objects as go
from plotly.subplots import make_subplots

root_folder = os.path.dirname(os.path.dirname( __file__ ))
results_folder = os.path.join(root_folder, 'results', 'clustering_models')

#--------------Table
db = SmellEvaluator('db')
tma = SmellEvaluator('tma')
im = SmellEvaluator('im')

#--------------Internal and External measurement figure

def resetIndex(df):
    df['newIndex'] = [i for i in range(1, (df.shape[0] + 1))]
    df = df.set_index(keys='newIndex', drop=True)
    return df

dfsInternal = {
    'db' : resetIndex(db.evalDf).head(5),
    'tma' : resetIndex(tma.evalDf).head(5),
    'im' : resetIndex(im.evalDf).head(5) 
}

colors = {
    'db' : '#0057e7',
    'tma' : '#d62d20',
    'im' : '#ffa700'
}

names = {
    'db' : 'Duplicate Block',
    'tma' : 'Too many Attributes',
    'im' : 'Insufficient Modularization'
}

def createTrace(smell, pm, legend):
    df = dfsInternal[smell]
    trace = dict(
        type = 'scatter',
        x = df.index,
        y = df[pm],
        mode = 'lines',
        line = dict(color = colors[smell], width=6),
        name = names[smell],
        showlegend = legend
    )
    return trace

fig = make_subplots(rows=6, cols=1, shared_xaxes=True, subplot_titles=['Silhouette Score (Higher is better)', 'Calinski-Harabasz Index (Higher is better)', 'Davies-Bouldin  Index (Lower is better)', 'Precision (Higher is better)', 'Matthews Correlation Coefficient (Higher is better)', 'Adjusted Rand Index (Higher is better)'], vertical_spacing = 0.05)

legend = True
for ix, pm in enumerate(['sc', 'ch', 'db', 'precision', 'mcc', 'ari']):
    ix += 1
    fig.append_trace(createTrace('db', pm, legend), ix, 1)
    fig.append_trace(createTrace('tma', pm, legend), ix, 1)
    fig.append_trace(createTrace('im', pm, legend), ix, 1)
    legend = False

fig.update_layout(
    height=2900, 
    width=2800, 
    title_text="Top 5 Configurations",
    paper_bgcolor='rgba(255, 255, 255, 1)',
    plot_bgcolor='rgba(255, 255, 255, 1)',
    legend=dict(x=-.1, y=1.2, orientation='h'),
    font = dict(size=47)
    )

#fix subplot title size
for i in fig['layout']['annotations']:
    i['font'] = dict(size=47)

#fig.show()
#fig.write_image(os.path.join(results_folder, 'configurationperformance.png'))


#-----------Stability
# (sparse, corr, out, pca, dista, algo)
dbTopConfigs = [
    (False, False, False, True, None, ('gm', 'spherical')),
    (False, False, True, True, None, ('gm', 'spherical')),
    (False, False, True, True, 'braycurtis', ('kmedoids', None)),
    (False, False, False, True, None, ('gm', 'tied')),
    (False, True, True, True, None, ('gm', 'full')),
]
 
tmaTopConfigs = [
    (False, False, False, True, None, ('gm', 'full')),
    (False, False, False, True, None, ('gm', 'tied')),
    (False, False, False, True, None, ('gm', 'spherical')),
    (False, False, True, True, None, ('gm', 'spherical')),
    (False, False, False, True, 'l1', ('agglo', 'complete'))
]


imTopConfigs = [
    (False, False, False, True, None, ('gm', 'full')),
    (False, False, False, True, None, ('gm', 'tied')),
    (False, False, True, True, None, ('gm', 'full')),
    (False, False, True, True, None, ('gm', 'spherical')),
    (False, False, False, True, None, ('gm', 'spherical'))
]

for ix, dbTopConfig in enumerate(dbTopConfigs):
    dbTopConfigModel = ClusterConfigurator(db.df, dbTopConfig)
    stability = dbTopConfigModel.getStability()
    print(f'DB config {ix}: {stability[0]}')

for ix, tmaTopConfig in enumerate(tmaTopConfigs):
    tmaTopConfigModel = ClusterConfigurator(tma.df, tmaTopConfig)
    stability = tmaTopConfigModel.getStability()
    print(f'TmA config {ix}: {stability[0]}')

for ix, imTopConfig in enumerate(imTopConfigs):
    imTopConfigModel = ClusterConfigurator(im.df, imTopConfig)
    stability = imTopConfigModel.getStability()
    print(f'IM config {ix}: {stability[0]}')




# tmaTopConfigModel = ClusterConfigurator(tma.df, tmaTopConfig)
# tmaStability = tmaTopConfigModel.getStability()

# imTopConfigModel = ClusterConfigurator(im.df, imTopConfig)
# imStability = imTopConfigModel.getStability()






#HIER DADELIJK NOG DE TOP ANALYSEREN EN OOK DE VERGELIJKING MET SCHWARZ MAKEN


# #Op deze manier kan je dan door een 
# topModel = SmellEvaluator('db').getPickle('db', test.evalDf.iloc[4].name)
# #Or if top one:
# topModel = SmellEvaluator('db').getPickle('db', test.topconfig.name)
# topModel.getStability()

#Als we dan de stability willen bereken moeten we m ff opnieuw aanroepen(nu tenminste, kan evt wel in loop)
#top_db = ClusterConfigurator()


#----------------old
#Only top 1
#df = df.head(10)

# for ix in range(0, 1):
#     print(df.iloc[ix]['name'], df.iloc[ix]['#clusterTrue'], df.iloc[ix]['#clusterFalse'])

# #Then, we calculate statistically test the clusters with eachother to find out which features are statistically different. 
# rejected_features = pd.Series()
# for ix in df['name']:
#     print(f'############## {ix} #############')
#     x_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'clustering', f'clusteringpipeline_smell-{smell}_{ix}'), 'rb')) 
#     dfTrue = x_df[x_df['cluster'] == True]
#     dfFalse = x_df[x_df['cluster'] == False]
#     dfTrue.drop(['cluster', smell], axis=1,  inplace=True)
#     dfFalse.drop(['cluster', smell], axis=1,  inplace=True)

#     sig_measurements = []
#     for measurement in dfTrue.columns:
#         print(f'------------------------{measurement}--------------------------')
#         try:
#             stat, pvalue = mannwhitneyu(dfTrue[measurement], dfFalse[measurement])
#             #print('P-value: ', pvalue)
#             if pvalue < 0.05:
#                 sig_measurements.append((measurement, pvalue))
#         except:
#             pass

#     print('# of Measurements: ', len(dfTrue.columns))
#     print('# of rejected Measurements: ', len(sig_measurements))

        # if pvalue < 0.05:
        #     fig = px.box(x_df, x='cluster', y=measurement, points='all')
        #     fig.write_image(os.path.join(results_folder, smell, f'{ix}_{measurement}.png'))
            #fig.show()




        #     print('True (smelly) distribution:')
        #     serie = dfTrue[measurement]
        #     zero_serie = serie[serie == 0]
        #     real_serie = serie[serie != 0]

        #     #MAKE FIGURE
        #     custom_font=dict(size=15)
        #     fig = make_subplots(
        #         rows=1,
        #         cols=2,
        #         specs=[[{}, {}]], 
        #         column_widths=[0.2, 0.8])


        #     fig.add_trace(
        #         go.Bar(
        #             x=list(zero_serie.value_counts().index),
        #             y=zero_serie.value_counts().tolist(),
        #             marker=dict(color='rgb(210,89,89)')
        #         ),
        #         row=1,
        #         col=1
        #     )

        #     fig.add_trace(
        #         go.Histogram(
        #             x=real_serie,
        #             xbins=dict(
        #                 start=np.min(real_serie),
        #                 end=np.max(real_serie),
        #                 size=(np.max(real_serie) - np.min(real_serie))/20
        #                 ),
        #                 marker=dict(color='rgb(0, 0, 100)')
        #         ),
        #         row=1,
        #         col=2
        #     )

        #     fig.update_layout(
        #         showlegend=False,
        #         bargap=0.2,
        #         paper_bgcolor='rgba(255, 255, 255, 1)', 
        #         plot_bgcolor='rgba(255, 255, 255, 1)'
        #     )

        #     fig.show()


        #     print('False (Non smelly) distribution:')
        #     serie = dfFalse[measurement]
        #     zero_serie = serie[serie == 0]
        #     real_serie = serie[serie != 0]

        #     #MAKE FIGURE
        #     custom_font=dict(size=15)
        #     fig = make_subplots(
        #         rows=1,
        #         cols=2,
        #         specs=[[{}, {}]], 
        #         column_widths=[0.2, 0.8])


        #     fig.add_trace(
        #         go.Bar(
        #             x=list(zero_serie.value_counts().index),
        #             y=zero_serie.value_counts().tolist(),
        #             marker=dict(color='rgb(210,89,89)')
        #         ),
        #         row=1,
        #         col=1
        #     )

        #     fig.add_trace(
        #         go.Histogram(
        #             x=real_serie,
        #             xbins=dict(
        #                 start=np.min(real_serie),
        #                 end=np.max(real_serie),
        #                 size=(np.max(real_serie) - np.min(real_serie))/20
        #                 ),
        #                 marker=dict(color='rgb(0, 0, 100)')
        #         ),
        #         row=1,
        #         col=2
        #     )

        #     fig.update_layout(
        #         showlegend=False,
        #         bargap=0.2,
        #         paper_bgcolor='rgba(255, 255, 255, 1)', 
        #         plot_bgcolor='rgba(255, 255, 255, 1)'
        #     )

        #     fig.show()


