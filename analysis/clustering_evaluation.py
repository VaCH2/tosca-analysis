# -nan eruit
# -sorteren per score
# - punten toekennen (beste krijgt 20, enabeste 18 en dat dan voor top 10 van true scores. voor silhouette beste 10, )
# - top 5 op basis van punten. 
# -balance uitrekenen. Ff bepalen wanneer een blans goed genoeg is.
# - Overview met scores en balans.
# - Bepaal hoeveel en welke measurements significant verschillen. Zijn dit er meer of minder dan in zn algemeen?


# Dan begint eik de manuele inspectie
# - Descriptives per optie bekijken (eerst ff alleen top voor elke smell)
# - Vergelijken met andere splits
# - Vragen uit afleiden die gevalideerd moeten worden door experts. 

import pickle
import os
import csv

from data import Data
from stats import Stats
from significance import Significance
from anomaly import AnomalyDetector
from utils import scale_df

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

 	
# The maximum width in characters of a column in the repr of a pandas data structure
pd.set_option('display.max_colwidth', -1)


root_folder = os.path.dirname(os.path.dirname( __file__ ))
results_folder = os.path.join(root_folder, 'results', 'clustering_models')

smell = 'alldummy'

scores = ['sil_score', 'dav_score', 'ch_score', 'ari_score', 'ami_score',
       'nmi_score', 'homogen_score', 'complete_score', 'v_score', 'fm_score']

internal_scores = ['sil_score', 'dav_score', 'ch_score']


df = pd.read_csv(
    os.path.join(results_folder, f'clustering_scores_{smell} - Copy.csv'),
    header=0)    

if smell is 'alldummy':
    df.drop(['ari_score', 'ami_score', 'nmi_score', 'homogen_score', 'complete_score', 'v_score', 'fm_score'], axis=1,  inplace=True)

df = df.dropna() 

df['total_score'] = 0
if smell is not 'alldummy':
    for score in scores:
        if score == 'dav_score':
            df = df.sort_values(by=score, ascending=True)
        else:
            df = df.sort_values(by=score, ascending=False)
        df = df.reset_index(drop=True)
        df['total_score'] = df['total_score'] + df.shape[0] - df.index.values
else:
    for score in internal_scores:
        if score == 'dav_score':
            df = df.sort_values(by=score, ascending=True)
        else:
            df = df.sort_values(by=score, ascending=False)
        df = df.reset_index(drop=True)
        df['total_score'] = df['total_score'] + df.shape[0] - df.index.values


#DIT NOG WEG
print('Option: ', df['name'].values[0])
print('Scores: ', df.iloc[0])
file_name = 'clusteringpipeline_smell-alldummy_excludeoutliers-False_excludecorr-False_pca-False_excludespars-True_2_None_gm_tied'
try_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'clustering', file_name), 'rb'))

def doeiets(ix, smell):
    #ix = "excludeoutliers-False_excludecorr-False_pca-True_None_('gm', 'tied')" #This one is for lr
    #ix = "excludeoutliers-True_excludecorr-False_pca-False_None_('gm', 'spherical')" #This one is for db

    file_name = f'clusteringpipeline_smell-{smell}_{ix}'
    option_df = pickle.load(open(os.path.join(root_folder, 'temp_data', 'clustering', file_name), 'rb'))
    return option_df['cluster'].value_counts()

if smell is not 'alldummy':
    df['#clusterTrue'] = df.apply (lambda row: doeiets(row['name'], smell)[True], axis=1)
    df['#clusterFalse'] = df.apply (lambda row: doeiets(row['name'], smell)[False], axis=1)

    print('True clusters: ', df['#clusterTrue'])
    print('False clusters: ', df['#clusterFalse'])

else:
    df['balancetop2'] = df.apply (lambda row: doeiets(row['name'], smell).iloc[1] / doeiets(row['name'], smell).iloc[0], axis=1)
    df = df[df['balancetop2'] > 0.2]

df = df.sort_values(by='total_score', ascending=False)

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


