from sklearn.preprocessing import StandardScaler
import copy


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