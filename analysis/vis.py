import pickle
import seaborn as sns
import matplotlib.pyplot as plt


distance = 'cosine'

df = pickle.load(open('../temp_data/dfpluscluster_{}'.format(distance), 'rb'))

df['cluster'] = df['cluster'].astype('category')


fig, ax = plt.subplots(10,5, figsize=(10,30))
important_metrics = df.columns
for ax, col in zip(ax.flat, important_metrics):
    if col != 'cluster':
        sns.boxplot(x="cluster", y=col, data=df, showfliers=False, ax=ax)
        ax.title.set_text(col)
        y_axis = ax.axes.get_yaxis()
        y_label = y_axis.get_label()
        y_label.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

plt.savefig('../temp_data/feature_boxplot_{}.png'.format(distance), dpi=300, bbox_inches='tight')
plt.show()