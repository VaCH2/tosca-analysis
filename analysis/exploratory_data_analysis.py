from classes.data import Data
from classes.stats import Stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import pickle

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

sns.set(style='white')
root_folder = os.path.dirname(os.path.dirname( __file__ ))
results_folder = os.path.join(root_folder, 'results', 'descriptive_report')

custom_font=dict(family="Open Sans", size=6, color="#7f7f7f")

data = Data()
df = data.dfs.get('all')

#Drop unused columns
df = df.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)
stats = Stats(df)

size_cols = ['loc_count', 'bloc_count', 'cloc_count', 'ntkn_count', 'nkeys_count', 'tett_count', 'tett_relative']

complex_cols = ['dpt_count', 'etp_count', 'nco_count', 'ninp_count', 
'ninpc_count', 'nout_count', 'nsh_count', 'nf_count', 'nac_count', 
'nfunc_count','nop_count', 'ntri_count' ]


sizecom_included_cols = [
    'na_count',
    'na_relative',
    'na_entropy',
    'nn_count',
    'nn_relative',
    'nn_entropy',
    'cdnt_count',
    'cdnt_entropy',
    'nc_count',
    'nrq_count',
    'np_count',
    'nif_count', 
]

sizecom_appendix_cols = [
    'nc_min',
    'nc_max',
    'nc_median',
    'nc_mean',
    'nif_min',
    'nif_max',
    'nif_median',
    'nif_mean',
    'np_min',
    'np_max',
    'np_median',
    'np_mean',
    'nrq_max',
    'nrq_median',
    'nrq_mean',
    'npol_count',
    'npol_relative',
    'npol_entropy',
    'nr_count',
    'nr_relative',
    'nr_entropy',
    'cdnt_relative',
    'cdrt_count',
    'cdrt_relative',
    'cdrt_entropy',
    'cdat_count',
    'cdat_relative',
    'cdat_entropy',
    'cdct_count',
    'cdct_relative',
    'cdct_entropy',
    'cddt_count',
    'cddt_relative',
    'cddt_entropy',
    'cdgt_count',
    'cdgt_relative',
    'cdgt_entropy',
    'cdit_count',
    'cdit_relative',
    'cdit_entropy',
    'cdpt_count',
    'cdpt_relative',
    'cdpt_entropy',
    'nw_count',
    'nw_relative',
    'nw_entropy',
    'nrq_min',
    'ngro_count',
    'ngro_relative',
    'ngro_entropy'
    ]


sizecom_cols = ['na_count',
'na_relative', 'na_entropy', 
'nc_count', 'nc_min', 
'nc_max', 'nc_median', 
'nc_mean', 'nif_count', 
'nif_min', 'nif_max',
'nif_median', 'nif_mean', 
'nn_count', 'nn_relative', 
'nn_entropy', 'np_count', 
'np_min', 'np_max', 
'np_median', 'np_mean', 
'nr_count', 'nr_relative',
'nr_entropy', 'cdnt_count', 
'cdnt_relative', 'cdnt_entropy', 
'cdrt_count', 'cdrt_relative', 
'cdrt_entropy', 'cdat_count', 
'cdat_relative', 'cdat_entropy', 
'cdct_count', 'cdct_relative', 
'cdct_entropy', 'cddt_count', 
'cddt_relative', 'cddt_entropy', 
'cdgt_count', 'cdgt_relative', 
'cdgt_entropy', 'cdit_count', 
'cdit_relative', 'cdit_entropy', 
'cdpt_count', 'cdpt_relative', 
'cdpt_entropy', 'nw_count', 
'nw_relative', 'nw_entropy', 
'nrq_count', 'nrq_min', 
'nrq_max', 'nrq_median', 
'nrq_mean', 'ngro_count', 
'ngro_relative', 'ngro_entropy', 
'npol_count', 'npol_relative', 
'npol_entropy']

other_cols = [ 'ni_count', 'ncys_count', 'noam_count', 'td_min', 
'td_max','td_median', 'td_mean', 'au_count',  'nscm_count' ]



#----------------------------------------------------------
# Files analysis

file_folder = os.path.join(root_folder, 'dataminer', 'tmp')
cleaned_ix = list(data.dfs.get('all').index)

files_dict = {}
users = os.listdir(file_folder)
for user in users:
    user_folder = os.path.join(file_folder, user)
    repos = os.listdir(user_folder)
    for repo in repos:
        identifier = os.path.join(user, repo)
        occurrences = len([ix for ix in cleaned_ix if identifier in ix])
        github_string = f'https://github.com/{user}/{repo}'
        files_dict[github_string] = occurrences

files_df = pd.DataFrame.from_dict(files_dict, orient='index')
#print(files_df.to_latex(index=True)) 
# for i, row in files_df.iterrows():
#     print('\href{', i, '}{', i.strip('https://'), '}', '&', str(row[0]), '\\')

#----------------------------------------------------------------
#Here the distribution plots for the splits

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{}, {}]]
)

fig.add_trace(
    go.Bar(
        x=list(Data('professionality').dfs.keys()),
        y=[value.shape[0] for value in Data('professionality').dfs.values()],
        marker=dict(color='rgb(0, 0, 100)')
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Bar(
        x=list(Data('purpose').dfs.keys()),
        y=[value.shape[0] for value in Data('purpose').dfs.values()],
        marker=dict(color='rgb(0, 0, 100)')
    ),
    row=1,
    col=2
)

fig.update_layout(
    height=600, 
    width=2551.2, 
    showlegend=False,
    bargap=0.2,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    paper_bgcolor='rgba(255, 255, 255, 1)',
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=35)
)

fig.show()
#fig.write_image(os.path.join(results_folder, 'data_distribution.png'))


#----------------------------------------------------------
# Stats table

sparsity = stats.totalsparsity

print(f'Sparsity for the data is: {sparsity}')

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

#This to get a filtered table on sparsity for readability
only_sizecom = metric_properties.loc[sizecom_cols, :]
only_sizecom_low_sparsity = only_sizecom[only_sizecom['sparsity'] < 0.895]
only_sizecom_high_sparsity = only_sizecom[only_sizecom['sparsity'] >= 0.895]

#Fix number of decimals and print to latex
for column in metric_properties.columns:
    for ix in metric_properties.index:
        value = metric_properties.loc[ix, column]
        if value % 1 == 0.0:
            metric_properties.loc[ix, column] = np.around(value)
        else:
            metric_properties.loc[ix, column] = np.around(value, decimals=2)

print(f'Metric properties \n {metric_properties}')
#metric_properties.to_csv(os.path.join(results_folder, 'stats.csv'), sep='\t', encoding='utf-8')

#Specificy columns you want!
metric_properties_per_category = metric_properties.loc[other_cols, :]
print(metric_properties_per_category.to_latex(index=True))

# #----------------------------------------------------------
# # Correlation matrix


to_exclude = ['nn_entropy', 'na_entropy', 'nr_relative', 'nr_entropy',
'nw_entropy', 'nw_relative', 'ngro_entropy', 'ngro_relative', 'npol_entropy', 'npol_relative',
'cdgt_entropy', 'cdgt_relative', 'cdrt_entropy', 'cdat_entropy', 'cdct_entropy',
'cddt_entropy', 'cdit_entropy', 'nif_min', 'nif_max', 'nif_mean', 'nif_median', 
'nc_median', 'np_median', 'nrq_median', 'td_median']

#INPUT!
# only_count_cols = [col for col in df.columns if '_count' in col or '_relative' in col or '_entropy' in col]
# df = df[only_count_cols]

filtered_df = df.drop(to_exclude, axis=1)
filtered_df.rename(columns=lambda x: f'{x.split("_")[0].upper()} {x.split("_")[1]}', inplace=True)
correlation_matrix = filtered_df.corr()

#Identify which combinations hold a certain correlation coefficient
for column in correlation_matrix.columns:
    for index, val in correlation_matrix[column].iteritems():
        if val > 0.7:
            if column != index:
                print(column, index, val)

for metric in correlation_matrix.columns:
    temp_matrix = correlation_matrix.copy()
    temp_matrix = temp_matrix.drop(metric)

    if (temp_matrix[metric] > 0.7).any() or (temp_matrix[metric] < -0.7).any():
        continue
    else:
        correlation_matrix = correlation_matrix.drop(metric, axis=1)
        correlation_matrix = correlation_matrix.drop(metric, axis=0)

    

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=correlation_matrix.as_matrix(),
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        xgap=10,
        ygap=10

    )
)

fig.update_layout(
    yaxis=dict(
        scaleanchor = "x",
        scaleratio = 1
    ),
    paper_bgcolor='rgba(255, 255, 255, 1)',
    plot_bgcolor='rgba(255, 255, 255, 1)',
    height=2551.2,
    width=2551.2,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=20
    ),
    font=dict(size=35) 
)

fig.update_xaxes(
            tickangle=45
        )

fig.show()
# fig.write_image(os.path.join(results_folder, 'alldata_correlationplot.png'))


#----------Heatmap with subplots

#INPUT!
relative_measurements = ['na_', 'nn_', 'nr_', 'cdnt_', 'cdrt_', 'cdat_', 'cdct_', 'cddt_', 'cdgt_', 
'cdit_', 'cdpt_', 'nw_', 'ngro_', 'npol_', 'tett_']

minmax_measurements = ['nc_', 'nif_', 'np_', 'nrq_', 'td_']

used_cols = relative_measurements

cols = 8
rows = 2 

fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles= tuple([col.strip('_').upper() for col in used_cols]),
    vertical_spacing = 0.1,
    shared_xaxes=True,
    shared_yaxes=True
)

for i, column in enumerate(used_cols):
    filtered_df = df[[col for col in df.columns if column in col]]
    filtered_df.rename(columns=lambda x: x.split(column)[1], inplace=True)
    correlation_matrix = filtered_df.corr()

    i += 1
    row = math.ceil(i / cols)
    if i % cols == 0:
        col = cols
    else:
        col = i % cols

    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.as_matrix(),
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            xgap=10,
            ygap=10,
        ),
        row=row,
        col=col
    )


#fix subplot title size
for i in fig['layout']['annotations']:
    i['font'] = dict(size=35)

fig.update_xaxes(
            tickangle=45
        )

fig.update_layout(
    height=703.2,  #For relative: 1479.7 which is 3 rows. or 703.2 in 2 rows. For minmax 576.6 which is 1 row
    width=2551.2, 
    showlegend=False,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=50
    ),
    paper_bgcolor='rgba(255, 255, 255, 1)', #transparant = 0,0,0,0
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=35) 
)

fig.show()


#----------------------------------------------------------
# Metric histograms

plot_cutoffs = {
    'loc_count': 1000,
    'bloc_count': 70,
    'cloc_count': 100,
    'dpt_count': None,
    'etp_count': None,
    'nco_count': None,
    'nkeys_count': 1000,
    'ntkn_count': 2000,
    'nscm_count': 10,
    'na_count': None,
    'na_relative' : None,
    'na_entropy' : None,
    'nc_count': None,
    'nc_min': None,
    'nc_max': None,
    'nc_median': None,
    'nc_mean': None,
    'ni_count': None,
    'nif_count': None,
    'nif_min' : None,
    'nif_max' : None,
    'nif_mean' : None,
    'nif_median' : None,
    'ninp_count': None,
    'ninpc_count': None,
    'nn_count': 50,
    'nn_relative' : None,
    'nn_entropy' : None,
    'nnt_count': None,
    'nout_count': None,
    'np_count': 150,
    'np_min': None,
    'np_max': None,
    'np_median': None,
    'np_mean': None,
    'nr_count': None,
    'nr_relative' : None,
    'nr_entropy' : None,
    'nrt_count': None,
    'cdnt_count': 20,
    'cdnt_relative': None,
    'cdnt_entropy': None,
    'cdrt_count': None,
    'cdrt_relative': None,
    'cdrt_entropy': None,
    'cdat_count': None,
    'cdat_relative': None,
    'cdat_entropy': None,
    'cdct_count': None,
    'cdct_relative': None,
    'cdct_entropy': None,
    'cddt_count': None,
    'cddt_relative': None,
    'cddt_entropy': None,
    'cdgt_count': None,
    'cdgt_relative': None,
    'cdgt_entropy': None,
    'cdit_count': None,
    'cdit_relative': None,
    'cdit_entropy': None,
    'cdpt_count': None,
    'cdpt_relative': None,
    'cdpt_entropy': None,
    'nw_count': None,
    'nw_relative': None,
    'nw_entropy': None,
    'nrq_count': None,
    'nrq_min': None,
    'nrq_max': None,
    'nrq_median': None,
    'nrq_mean': None,
    'nsh_count': None,
    'ncys_count': None,
    'ngro_count': None,
    'ngro_relative': None,
    'ngro_entropy': None,
    'npol_count': None,
    'npol_relative': None,
    'npol_entropy': None,
    'nf_count': None,
    'td_min' : None,
    'td_max' : None,
    'td_mean' : None,
    'td_median' : None,
    'au_count' : None,
    'nac_count' : None,
    'nfunc_count' : 100,
    'noam_count' : 100,
    'nop_count' : None,
    'ntri_count' : None,
    'tett_count' : 250,
    'tett_relative' : None,
    
}
#INPUT!
df = df[complex_cols]

new_df = pd.DataFrame(index=df.index)
#This loop to split columns in zero en nonzero
for column in df.columns:

    #Filter cutoff
    if plot_cutoffs[column] != None:
        df[column] = df[column].where(df[column] < plot_cutoffs[column])

    new_df[f'{column}_zero'] = df[column][df[column] == 0]
    new_df[f'{column}_real'] = df[column][df[column] != 0]

df = new_df

#MAKE FIGURE

custom_font=dict(size=35)

cols = 11
rows = 3 #Rows zelf invullen size=2, com=3, sizecom_incl=3, sizecom_app=13, other=3
fig = make_subplots(
    rows=rows,
    cols=cols,
    
    #REPLACE ZERO columns with ''
    subplot_titles= tuple([name.split('_real')[0] if 'zero' not in name else ' ' for name in df.columns]),

    #listcomprehension for number of rows
    specs=[ [{}, {}, None, {}, {}, None, {}, {}, None, {}, {}] for row in range(rows)],
    column_widths=[0.0287, 0.1763, 0.06, 0.0287, 0.1763, 0.06, 0.0287, 0.1763, 0.06, 0.0287, 0.1763]
)

not_in1 = list(range(2, 1000, 11))
not_in2 = list(range(5, 1000, 11))
not_in3 = list(range(8, 1000, 11))
skip_ixs = not_in1 + not_in2 + not_in3

ix_col = 0

for i, column in enumerate(df.columns):

    if ix_col not in skip_ixs:
        ix_col += 1
    else:
        ix_col += 2

    row = math.ceil(ix_col / cols)

    if ix_col % cols == 0:
        col = cols
    else:
        col = ix_col % cols

    counts = df[column]

    value_counts = counts.value_counts().sort_index()

    if 'zero' in column:
        fig.add_trace(
            go.Bar(
                x=list(value_counts.index),
                y=value_counts.tolist(),
                marker=dict(color='rgb(210,89,89)')
            ),
            row=row,
            col=col
        )
        if len(value_counts) != 0:
            fig.update_yaxes(
                row=row,
                col=col,
                range=[0, 1105],
                dtick=value_counts[0],
                nticks=1
            )
        else:
            fig.update_yaxes(
                row=row,
                col=col,
                visible=False
            )

        fig.update_xaxes(
            row=row,
            col=col,
            tick0=0,
            nticks=1
        )

    elif len(value_counts) < 6:
        fig.add_trace(
            go.Bar(
                x=list(value_counts.index),
                y=value_counts.tolist(),
                marker=dict(color='rgb(0, 0, 100)')
            ),
        row=row, 
        col=col
        )

    #If there are floats (mean metric for example)
    elif sum([ix % 1 for ix in value_counts.index]) != 0:
        fig.add_trace(
            go.Histogram(
                x=counts,
                xbins=dict(
                    start=np.min(counts),
                    end=np.max(counts),
                    size=(np.max(counts) - np.min(counts))/20
                ),
                marker=dict(color='rgb(0, 0, 100)')
            ),
        row=row, 
        col=col
        )

        fig.update_yaxes(
            row=row,
            col=col,
            nticks=4
        )

        fig.update_xaxes(
            row=row,
            col=col,
            tickangle=45
        )

    else:
        fig.add_trace(
            go.Histogram(
                x=counts,
                xbins=dict(
                    start=np.min(counts),
                    end=np.max(counts),
                    size=1
                ),
                marker=dict(color='rgb(0, 0, 100)')
            ),
        row=row, 
        col=col
        )
        
        fig.update_yaxes(
            row=row,
            col=col,
            range=[0, np.max(value_counts)],
            tick0=0,
            dtick=(int(np.max(value_counts)/3))

        )

        x_ax_values = value_counts.index.values
        x_ax_values.sort()
        third_last_element = x_ax_values[-2] 
        fig.update_xaxes(
            row=row,
            col=col,
            range=[0, third_last_element],
            tick0=1,
            dtick=(int(third_last_element/3))
        )

#fix subplot title size
for i in fig['layout']['annotations']:
    i['font'] = dict(size=35)


fig.update_layout(
    height=(50-30+130*rows)*5, 
    width=510.236221*5, 
    showlegend=False,
    bargap=0.2,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=50
    ),
    paper_bgcolor='rgba(255, 255, 255, 1)', #transparant = 0,0,0,0
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=custom_font
)

fig.show()
# fig.write_image(os.path.join(results_folder, 'alldata_histograms.png'))


#-----------------SMELL ANALYSIS---------------

#--------table------------

smells = pd.read_excel('../results/labeling/to_label.xlsx', sheet_name='Sheet1', usecols='B:H', nrows=685, index_col=0)

#Professionalism
for smell in ['db', 'tma', 'im']:
    for subset in ['Example', 'Industry']:
        df = Data('professionality').dfs.get(subset)
        df = df.merge(smells[smell], how='left', left_index=True, right_index=True)
        counts = df[smell].value_counts()
        print(smell, subset, ' absolute: ', counts.loc[1])
        rel = (counts.loc[1] /(counts.loc[0] + counts.loc[1]))*100
        print(smell, subset, ' relative: ', rel)
        print(smell, subset, ' SIZE: ', (counts.loc[0] + counts.loc[1]))

#Purpose
for smell in ['db', 'tma', 'im']:
    for subset in ['topology', 'custom', 'both', 'none']:
        df = Data('purpose').dfs.get(subset)
        df = df.merge(smells[smell], how='left', left_index=True, right_index=True)
        counts = df[smell].value_counts()
        print(smell, subset, ' absolute: ', counts.loc[1])
        rel = (counts.loc[1] /(counts.loc[0] + counts.loc[1]))*100
        print(smell, subset, ' relative: ', rel)
        print(smell, subset, ' SIZE: ', (counts.loc[0] + counts.loc[1]))

#Repository
for smell in ['db', 'tma', 'im']:
    for subset in ['openstack-tosca-parser', 'radon-h2020-radon-particles', 'ystia-forge', 'alien4cloud-csar-public-library', 'tliron-puccini']:
        df = Data('repo').dfs.get(subset)
        df = df.merge(smells[smell], how='left', left_index=True, right_index=True)
        counts = df[smell].value_counts()
        print(smell, subset, ' absolute: ', counts.loc[1])
        rel = (counts.loc[1] /(counts.loc[0] + counts.loc[1]))*100
        print(smell, subset, ' relative: ', rel)
        print(smell, subset, ' SIZE: ', (counts.loc[0] + counts.loc[1]))


fig = make_subplots(
    rows=1,
    cols=6,
    specs=[[{}, {},{}, {},{}, {}]],
    shared_yaxes=True,
    subplot_titles = ('Long Statement', 'Too Many Attributes', 'Duplicate Block', 'Long Resource', 'Insufficient Modularization', 'Weakened Modularity'),
    vertical_spacing=6
)

for count, smell in enumerate(['ls', 'db', 'lr', 'tma', 'im', 'wm']):
    count += 1

    if smell in ['db', 'tma', 'im']:

        fig.add_trace(
            go.Bar(
                x=smells[smell].value_counts().index.values,
                y=smells[smell].value_counts(),
                marker=dict(color=['rgb(210,89,89)', 'rgb(0, 0, 100)'])
            ),
            row=1,
            col=count
        )
    
    else:
        fig.add_trace(
            go.Bar(
                x=smells[smell].value_counts().index.values,
                y=smells[smell].value_counts(),
                marker=dict(color=['rgb(255, 207, 193)', 'rgb(157, 119, 236)'])
            ),
            row=1,
            col=count
        )




fig.update_layout(
    height=1000, 
    width=3500.2, 
    showlegend=False,
    bargap=0.1,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=50
    ),
    paper_bgcolor='rgba(255, 255, 255, 1)', #transparant = 0,0,0,0
    plot_bgcolor='rgba(255, 255, 255, 1)',
    font=dict(size=35)
)

fig.update_traces(texttemplate='%{y}', textposition='outside')

for i in fig['layout']['annotations']:
    i['font'] = dict(size=40)

fig.update_yaxes(
    showgrid=True,
    gridwidth=5,
    gridcolor='rgba(255, 255, 255, 1)'
)

fig.show()
# fig.write_image(os.path.join(results_folder, 'smell_distribution.png'))