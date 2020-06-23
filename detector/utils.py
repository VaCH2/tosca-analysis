from sklearn.preprocessing import StandardScaler
import copy
import numpy as np



def scale_df(df):
    copy_df = df.copy(deep=True)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(copy_df)
    copy_df.loc[:,:] = scaled_values
    return copy_df