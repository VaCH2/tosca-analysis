from data import Data

df = Data().dfs.get('all')

sample_df = df.sample(285, random_state=1)



sample_df.to_excel('to_label.xlsx')