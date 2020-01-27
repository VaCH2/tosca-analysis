import os
import pandas as pd
from toscametrics import calculator
import pickle

class Data():
    def __init__(self, metrics_type, dataset, file_type='all'):
        datasets_dir = {
            'all'       :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\1. All',
            'industry'  :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\1. All\\Total Industry',
            'example'   :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\1. All\\Total Examples',
            'repos'     :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\2. Repositories',
            'a4c'       :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\2. Repositories\\A4C',
            'forge'     :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\2. Repositories\\Forge',
            'puccini'   :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\2. Repositories\\Puccini'
        }

        file_types = ['all', 'topology', 'custom', 'both', 'none']

        if not dataset in datasets_dir.keys():
            raise ValueError('Enter a valid dataset (all, industry, example, repos, a4c, forge, puccini)')

        if not file_type in file_types:
            raise ValueError('Enter a valid file type (all, topology, custom, both, none)')

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

        self.typefilterpercentage = None

        if file_type != 'all':
            self.typefilterpercentage = self.filter_filetype(file_type)

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
        df = df.dropna()
        cols = df.select_dtypes(include=['bool', 'object']).columns
        df[cols] = df[cols].astype(int)
        df = df.dropna()
        return df

    
    def filter_filetype(self, filetype):
        '''Filter on the file type. A file could be a service template, or containing
        custom type definitions, both or none of these two. It assigns the filtered df
        to self.df and assings the filtered percentage to typefilterpercentage'''

        if not filetype in ['topology', 'custom', 'both', 'none']:
            raise ValueError('Enter a valid file type (topology, custom, both, none)')

        cus_df = self.df[['cdnt_count', 'cdrt_count', 'cdat_count', 
        'cdct_count', 'cddt_count', 'cdgt_count', 'cdit_count', 'cdpt_count']]

        non_df = cus_df[(cus_df == 0).all(1)] 
        self.df['custom_def'] = [False if x in non_df.index else True for x in self.df.index]

        if filetype == 'topology':
            result = self.df[(self.df['ttb_check'] == 1) & (self.df['custom_def'] == False)]

        if filetype == 'custom':
            result = self.df[(self.df['ttb_check'] == 0) & (self.df['custom_def'] == True)]

        if filetype == 'both':
            result = self.df[(self.df['ttb_check'] == 1) & (self.df['custom_def'] == True)]

        if filetype == 'none':
            result = self.df[(self.df['ttb_check'] == 0) & (self.df['custom_def'] == False)]

        result = result.drop('custom_def', axis=1)
        self.df = result
        return len(result)/len(cus_df)        
    