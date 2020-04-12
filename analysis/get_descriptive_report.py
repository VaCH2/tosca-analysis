from data import Data
from stats import Stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

from plotly.subplots import make_subplots
import plotly.graph_objects as go

sns.set(style='white')
root_folder = os.path.dirname(os.path.dirname( __file__ ))
results_folder = os.path.join(root_folder, 'results', 'descriptive_report')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def main():
    data = Data('all')
    df = data.dfs.get('all')
    stats = Stats(data.dfs.get('all'))

#----------------------------------------------------------
# Stats table

    # sparsity = stats.totalsparsity
    # print(f'Sparsity for the data is: {sparsity}')


    # min = stats.min
    # max = stats.max
    # nonzero = stats.nonzero
    # zero = stats.zero
    # sparsity = stats.featuresparsity
    # mean = stats.mean
    # stddv = stats.stddv
    # q1 = stats.q1
    # median = stats.median
    # q3 = stats.q3
    # metric_properties = pd.concat([min, max, nonzero, zero, sparsity, mean, stddv, q1, median, q3], axis=1)
    # print(f'Metric properties \n {metric_properties}')

#----------------------------------------------------------
# Correlation matrix
#Moeten wel minder waardes in. Wellicht filteren op degene met
#een correlatie boven de 0.5 of onder de -0.5 GEDAAN

    #Blue, maar werkt niet.. te kut met die verschillende schalen
    # colorscale = [
    #     [0, '#000064'],
    #     [0.1, '#000064'],
    #     [0.1, '#CCCCE0'],
    #     [0.25, '#CCCCE0'],
    #     [0.25, '#FFFFFF'],
    #     [0.75, '#FFFFFF'],
    #     [0.75, '#CCCCE0'],
    #     [0.9, '#CCCCE0'],
    #     [0.9, '#000064'],
    #     [1, '#000064']
    # ]
    
    #Find highly correlating metrics
    correlation_matrix = df.corr()

    for metric in correlation_matrix.columns:
        temp_matrix = correlation_matrix.copy()
        temp_matrix = temp_matrix.drop(metric)

        if (temp_matrix[metric] > 0.8).any() or (temp_matrix[metric] < -0.8).any():
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
            #colorscale=colorscale,
            zmid=0,
            # colorbar=dict(
            #     tickmode='linear'#,
                #nticks = 6
            #     tickvals=[0, 0.1, 0.25, 0.75, 0.9, 1],
            #     ticktext=['-1', '-0.8', '-0.5', '0.5', '0.8', '1']
            #     #tick0=-1
            #     # dtick=1
            # ),
            #xtype='scaled',
            #ytype='scaled',
            # x0=0,
            # y0=0,
            # dx=5,
            # dy=5,
            xgap=3,
            ygap=3

        )
    )

    fig.update_layout(
        yaxis=dict(
            scaleanchor = "x",
            scaleratio = 1,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=510.236221,
        width=510.236221,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=20
        )
    )

    fig.show()
    fig.write_image(os.path.join(results_folder, 'alldata_correlationplot.png'))


    
    #----------------------------------------------------------
    # Metric histograms

    # plot_cutoffs = {
    #     'loc_count': 1000,
    #     'bloc_count': None,
    #     'cloc_count': None,
    #     'dpt_count': None,
    #     'etp_count': None,
    #     'nco_count': 15,
    #     'nkeys_count': None,
    #     'ntkn_count': None,
    #     'nscm_count': None,
    #     'na_count': None,
    #     'nc_count': None,
    #     'nc_min': None,
    #     'nc_max': None,
    #     'nc_median': None,
    #     'nc_mean': None,
    #     'ni_count': None,
    #     'nif_count': None,
    #     'ninp_count': None,
    #     'ninpc_count': None,
    #     'nn_count': None,
    #     'nnt_count': None,
    #     'nout_count': None,
    #     'np_count': None,
    #     'np_min': None,
    #     'np_max': None,
    #     'np_median': None,
    #     'np_mean': None,
    #     'nr_count': None,
    #     'nrt_count': None,
    #     'ttb_check': None,
    #     'cdnt_count': None,
    #     'cdrt_count': None,
    #     'cdat_count': None,
    #     'cdct_count': None,
    #     'cddt_count': None,
    #     'cdgt_count': None,
    #     'cdit_count': None,
    #     'cdpt_count': None,
    #     'nw_count': None,
    #     'tdb_check': None,
    #     'nrq_count': None,
    #     'nsh_count': None,
    #     'ncys_count': None,
    #     'tob_check': None,
    #     'ngc_count': None,
    #     'ngp_count': None,
    #     'ngro_count': None,
    #     'npol_count': None,
    #     'nf_count': None
    # }

    # cols = 3
    # rows = math.ceil(len(df.columns)/cols)
    # fig = make_subplots(
    #     rows=rows,
    #     cols=cols,
    #     #subplot_titles= tuple([metric.split('_')[0] for metric in df.columns])
    #     subplot_titles= tuple([metric for metric in df.columns])
    # )

    # for i, column in enumerate(df.columns):
    #     i += 1
    #     row = math.ceil(i / cols)

    #     if i % cols == 0:
    #         col = cols
    #     else:
    #         col = i % cols

    #     counts = df[column]
    #     if plot_cutoffs[column] != None:
    #         counts = counts[counts <= plot_cutoffs[column]]
        
    #     #counts = counts.tolist()
    #     value_counts = counts.value_counts().sort_index()

    #     if len(value_counts) < 6:
    #         fig.add_trace(
    #             go.Bar(
    #                 x=list(value_counts.index),
    #                 y=value_counts.tolist(),
    #                 marker=dict(color='rgb(0, 0, 100)')
    #             ),
    #         row=row, 
    #         col=col
    #         )

    #     #If there are floats (mean metric for example)
    #     elif sum([ix % 1 for ix in value_counts.index]) != 0:
    #         fig.add_trace(
    #             go.Histogram(
    #                 x=counts,
    #                 xbins=dict(
    #                     start=np.min(counts),
    #                     end=np.max(counts),
    #                     size=0.5
    #                 ),
    #                 marker=dict(color='rgb(0, 0, 100)')
    #             ),
    #         row=row, 
    #         col=col
    #         )

    #     else:
    #         fig.add_trace(
    #             go.Histogram(
    #                 x=counts,
    #                 xbins=dict(
    #                     start=np.min(counts),
    #                     end=np.max(counts),
    #                     size=1
    #                 ),
    #                 marker=dict(color='rgb(0, 0, 100)')
    #             ),
    #         row=row, 
    #         col=col
    #         )
        
    #     #To  make all subplots log (Doesnt look nice)
    #     # fig.update_yaxes(
    #     #     type="log", 
    #     #     row=row, 
    #     #     col=col
    #     # )
    
    # fig.update_layout(
    #     height=3000, 
    #     width=510.236221, 
    #     showlegend=False,
    #     bargap=0.2,
    #     margin=dict(
    #         l=0,
    #         r=0,
    #         b=0,
    #         t=20
    #     )
    # )
    
    # fig.show()
    # fig.write_image(os.path.join(results_folder, 'alldata_histograms.png'))





if __name__=='__main__':
    main()