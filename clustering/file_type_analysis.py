#%%
df = Dataset(1, 'all').getDf
#%%
cus_df = df[['cdnt_count', 'cdrt_count', 'cdat_count', 
'cdct_count', 'cddt_count', 'cdgt_count', 'cdit_count', 'cdpt_count']]

non_df = cus_df[(cus_df == 0).all(1)] 
df['custom_def'] = [False if x in non_df.index else True for x in df.index]


# %%
top_df = df[(df['ttb_check'] == 1) & (df['custom_def'] == False)]
cus_df = df[(df['ttb_check'] == 0) & (df['custom_def'] == True)]
both_df = df[(df['ttb_check'] == 1) & (df['custom_def'] == True)]
non_df = df[(df['ttb_check'] == 0) & (df['custom_def'] == False)]

print('topology: ', len(top_df))
print('custom definition: ', len(cus_df))
print('both: ',len(both_df))
print('none: ',len(non_df))


# %%
