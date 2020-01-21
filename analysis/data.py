import os
import pandas as pd
from toscametrics import calculator
import pickle

class Data():
    def __init__(self, metrics_type, dataset):
        datasets_dir = {
            'all'       :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\4. All',
            'industry'  :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\2. Total Industry',
            'example'   :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\1. Total Examples',
            'a4c'       :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\A4C',
            'forge'     :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\Forge',
            'puccini'   :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\Puccini'
        }
        if not dataset in datasets_dir.keys():
            raise ValueError('Enter a valid dataset (all, industry, example, a4c, forge, puccini)')

        try:
            self.raw_df = pickle.load(open('../temp_data/{}_{}_raw_df'.format(metrics_type, dataset), 'rb'))
            
        except (OSError, IOError) as e:
            files = self.get_yaml_files(datasets_dir[dataset])
            json_data = self.json_data(metrics_type, files)
            self.raw_df = self.to_df(json_data)
            pickle.dump(self.raw_df, open('../temp_data/{}_{}_raw_df'.format(metrics_type, dataset), 'wb'))

        try:
            self.df = pickle.load(open('../temp_data/{}_{}_df'.format(metrics_type, dataset), 'rb'))
        
        except (OSError, IOError) as e:
            self.df = self.cleaning(self.raw_df)
            pickle.dump(self.df, open('../temp_data/{}_{}_df'.format(metrics_type, dataset), 'wb'))


    def get_yaml_files(self, path):
        extensions = ['.yaml', '.yml']
        allFiles = []

        listOfFile = os.listdir(path)

        for entry in listOfFile:
            fullPath = os.path.join(path, entry)
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.get_yaml_files(fullPath)
            else: 
                for extension in extensions:
                    if fullPath.endswith(extension):
                        allFiles.append(fullPath)       
        
        return allFiles

    def json_data(self, metrics_type, yaml_files):
        metrics = calculator.MetricCalculator(yaml_files, metrics_type).getresults
        return metrics

    def to_df(self, json_data):
        #Transform JSON file to Pandas DataFrame
        flat_dict = {}
        for key, value in json_data.items():
            df = pd.io.json.json_normalize(value, sep='_')
            value = df.to_dict(orient='records')[0]
            flat_dict[key] = value

        df = pd.DataFrame.from_dict(flat_dict, orient='index')
        return df

    def cleaning(self, df):        
        #Drop NaN rows and error columns, and make numeric
        df = df.drop(labels=(df.filter(regex='msg').columns), axis=1)
        cols = df.select_dtypes(include=['bool']).columns
        df[cols] = df[cols].astype(int)
        df = df.dropna()
        return df        
    