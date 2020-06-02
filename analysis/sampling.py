import pandas as pd
from data import Data

all = Data().dfs.get('all')
df = pd.read_excel('../results/labeling/to_label.xlsx', sheet_name='Sheet1')

toExclude = df['Blueprint path'].dropna()
all = all.drop(toExclude)

spl = all.sample(n=400, random_state=1)
spl = spl.reset_index()
spl['index'].to_excel("output.xlsx")