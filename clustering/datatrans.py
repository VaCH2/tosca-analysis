import pandas as pd
import json


class Dataset():

    def __init__(self, version, example=False):
        """Initialize a new dataset containing industry files by default. 
        Version defines the version of the calculated metrics."""

        try:
            if example == False:
                with open('..\\tosca-metrics\\results\\industry_metric_results_{}.json'.format(version)) as f:
                    self.data = json.load(f)

            if example == True:
                with open('..\\tosca-metrics\\results\\example_metric_results_{}.json'.format(version)) as f:
                    self.data = json.load(f)

            self.cleaned_data = self.__cleaning()   
                     
        except:
            raise


    def __cleaning(self):
        #Transform JSON file to Pandas DataFrame
        flat_dict = {}
        for key, value in self.data.items():
            df = pd.io.json.json_normalize(value, sep='_')
            value = df.to_dict(orient='records')[0]
            flat_dict[key] = value

        df = pd.DataFrame.from_dict(flat_dict, orient='index')
        
        #Drop NaN rows, columns with a constant value, Error message or a high correlation   
        df = df.drop(labels=(df.filter(regex='msg').columns), axis=1)
        rejected = ['nc_max', 'nc_mean', 'nc_min', 'ngro_count', 'nkeys_count', 'np_count', 
        'np_max', 'np_mean', 'np_median', 'np_min', 'npol_count', 'nr_count', 'nrt_count', 'nw_count']
        df = df.drop(labels=rejected, axis=1)
        df = df.dropna()
        
        #Make all columns numeric
        df.ttb_check = df.ttb_check.astype(int)
        df.tdb_check = df.tdb_check.astype(int)
        df.tob_check = df.tob_check.astype(int)
        return df

    @property
    def getDf(self):
        return self.cleaned_data