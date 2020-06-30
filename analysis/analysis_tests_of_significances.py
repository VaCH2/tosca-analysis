from data import Data
from stats import Stats
from significance import Significance
import itertools as it
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

root_folder = os.path.dirname(os.path.dirname( __file__ ))
results_folder = os.path.join(root_folder, 'results', 'descriptive_report')

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

def get_rejections(split, threshold, discardzeroes):
    data = Data(split)
    keys = list(data.dfs.keys())

    columns = Data('all').dfs.get('all')
    columns = columns.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
    if split == 'purpose':
        columns = columns.drop([key for key in columns.columns if 'cd' in key], axis=1)
    columns = columns.columns

    scores = {column : [] for column in columns}
    df_pvalues = pd.DataFrame(index=columns)

    for a,b in it.combinations(keys, 2):
        df1 = data.dfs[a]
        df2 = data.dfs[b]

        if split == 'purpose':
            df1 = df1.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
            df2 = df2.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
            df1 = df1.drop([key for key in df1.columns if 'cd' in key], axis=1)
            df2 = df2.drop([key for key in df2.columns if 'cd' in key], axis=1)
        else:
            df1 = df1.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
            df2 = df2.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)

        #Extra iteration to take only blueprints with nodes
        if split == 'professionality':
            df1 = df1[df1['nn_count'] != 0]
            df2 = df2[df2['nn_count'] != 0]

        sig_analysis = Significance(df1, df2, discardzeroes)
        try:
            stat_values = sig_analysis.sig['corr_p_values']
            df_pvalues = pd.concat([df_pvalues, stat_values], axis=1)
        except:
            continue

        for column in columns:
            try:
                scores[column].append(sig_analysis.sig['rejected'].loc[column])
            except:
                pass

        #print rejections for purpose to describe results in thesis
        if split == 'purpose':
            print(f'combination: {a}-{b}')
            print(sig_analysis.sig['rejected'])

    rejection_count = {}
    for key in columns:
        rejection_count[key] = scores[key].count(True)

    df_rejection_count = pd.DataFrame.from_dict(rejection_count, orient='index', columns=['count'])
    return df_rejection_count.sort_index(), df_pvalues.sort_index()



def stats_per_split(split, sig_ix):
    data = Data(split).dfs
    for key, value in data.items():

        #value = value[value['nn_count'] != 0]
        test = ['alien4cloud-csar-public-library', 'openstack-tosca-parser', 'radon-h2020-radon-particles', 'tliron-puccini', 'ystia-forge']
        if key in test:
            stats = Stats(value)
            min = stats.min
            max = stats.max
            nonzero = stats.nonzero
            zero = stats.zero
            sparsity = stats.featuresparsity
            mean = stats.mean
            stddv = stats.stddv
            q1 = stats.q1
            median = stats.median
            q3 = stats.q3
            metric_properties = pd.concat([min, max, nonzero, zero, sparsity, mean, stddv, q1, median, q3], axis=1)
            metric_properties  = metric_properties[metric_properties.index.isin(sig_ix)].sort_index()
            metric_properties.to_excel(os.path.join(results_folder, f'{key}_stats.xlsx'))


ranges = {
    'professionality' : [0, 7],
    'purpose' : [0,7],
    'repo' : [0,95]
}



splits = ['professionality', 'purpose', 'repo']
df_pvalues = pd.DataFrame(index=Data('all').dfs.get('all').columns)
#df_pvalues.to_excel(os.path.join(results_folder, f'splits_rejectionpvalues.xlsx'))

for split in splits:

    rejection_count, pvalues = get_rejections(split, 0, False)
    df_pvalues = pd.concat([df_pvalues, pvalues.mean(axis=1)], axis=1)

    #Create stats per split
    if split != 'repo':
        stats_per_split(split, rejection_count.index.values)

    #rejection_count = rejection_count.sort_values(by=['count'], ascending=False)
    rejection_count.rename(index=lambda x: f'{x.split("_")[0].upper()} {x.split("_")[1]}', inplace=True)
    rejection_count.replace(0, 0.24, inplace=True)
    rejection_count = rejection_count.iloc[::-1]

    
    fig = go.Figure(
        go.Bar(
            x=rejection_count['count'],
            y=rejection_count.index.values,
            marker=dict(
                color='rgb(255, 255, 255)',
                line=dict(color='rgb(0, 0, 0)', width=3)),
            orientation='h'
        )
    )
    
    fig.update_traces(
        texttemplate='%{x:.0f}', 
        textposition='outside',
        outsidetextfont = dict(size=40), 
        cliponaxis=False
        )

    fig.update_layout(
        xaxis= dict(
            range=ranges[split]
        ),
        width=1000,
        height=6000,
        paper_bgcolor='rgba(255, 255, 255, 1)', 
        plot_bgcolor='rgba(255, 255, 255, 1)',
        font=dict(size=40),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=30
        ),
    )
    
    fig.show()
    # fig.write_image(os.path.join(results_folder, f'{split}_rejectionbar.png'))