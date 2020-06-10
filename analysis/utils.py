from sklearn.preprocessing import StandardScaler
import copy
import numpy as np


def scale_df(df):
    copy_df = df.copy(deep=True)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(copy_df)
    copy_df.loc[:,:] = scaled_values
    return copy_df

def flatlist(nested_list):
    return [item for sublist in nested_list for item in sublist]

def allin2(nested_list):
    all = set()

    for element in flatlist(nested_list):
        if element in nested_list[0]:
            if element in nested_list[1]:
                all.add(element)
    return list(all)

def allin3(nested_list):
    all = set()

    for element in flatlist(nested_list):
        if element in nested_list[0]:
            if element in nested_list[1]:
                if element in nested_list[2]:
                    all.add(element)
    return list(all)

def df_minus_df(df1, df2):
    df = copy.deepcopy(df1)
    ixs = [ix for ix in list(df.index) if ix not in list(df2.index)]
    df = df.loc[ixs]

    return df


# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s