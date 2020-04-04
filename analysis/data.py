import os
import pandas as pd
from toscametrics import calculator
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import df_minus_df

class Data():
    #desnoods die metrics type hierin code ipv in die calculator en dan gewoon die hardcoden op 'all'
    def __init__(self, split='all', metrics_type='tosca_and_general'):

        try:
            raw_df = pickle.load(open('../temp_data/all_raw_df', 'rb'))
            
        except (OSError, IOError):
            files = self.get_indices('all', None)
            json_data = self.json_data(metrics_type, files.get('all'))
            raw_df = self.to_df(json_data)
            pickle.dump(raw_df, open('../temp_data/all_raw_df', 'wb'))


        self.raw_df = raw_df
        raw_size = self.raw_df.shape[0]


        try:
            df = pickle.load(open('../temp_data/all_df', 'rb'))

        except (OSError, IOError):
            df = self.cleaning(self.raw_df)
            pickle.dump(df, open('../temp_data/all_df', 'wb'))


        cleaned_size = df.shape[0]
        self.droppedrows = raw_size - cleaned_size
        

        split_indices = self.get_indices(split, df)
        self.dfs = {split_element : df.loc[indices] for split_element, indices in split_indices.items()}
        

    def get_indices(self, split, df):
        data_path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\Repositories'
        repos = os.listdir(data_path)
        
        professionalities = ['Example', 'Industry']

        if split == 'repo':
            split_paths ={repo : [data_path + r'\{}'.format(repo)] for repo in repos}

        elif split == 'professionality':
            repo_paths ={repo : data_path + r'\{}'.format(repo) for repo in repos}
            split_paths = {}
            
            for prof in professionalities:
                split_paths[prof] = [repo_path + r'\{}'.format(prof) for repo_path in repo_paths.values()]


        elif split == 'purpose':
            split_files = self.filter_filetype(df)

        elif split == 'all':
            split_paths = {'all' : [data_path]}

        else:
            raise ValueError

        if split != 'purpose':
            split_files = {}
            for split, paths in split_paths.items():
                
                files = []
                for path in paths:
                    files.extend(self.get_yaml_files(path))
                
                #make sure dropped files are not included
                files = [file for file in files if file in list(df.index)]
                split_files[split] = files
        
        return split_files





        # datasets_dir = {
        #     'all'       :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\All',
        #     'industry'  :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\Total Industry',
        #     'example'   :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\Total Examples',
        #     'repos'     :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\Repositories',
        #     'a4c'       :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\Repositories\A4C',
        #     'forge'     :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\Repositories\Forge',
        #     'puccini'   :   r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\Repositories\Puccini'
        # }

        # purpose_types = ['all', 'topology', 'custom', 'both', 'none']

        # if not dataset in datasets_dir.keys():
        #     raise ValueError('Enter a valid dataset (all, industry, example, repos, a4c, forge, puccini)')

        # if not file_type in file_types:
        #     raise ValueError('Enter a valid file type (all, topology, custom, both, none)')

        # try:
        #     self.raw_df = pickle.load(open('../temp_data/{}_{}_raw_df'.format(metrics_type, dataset), 'rb'))
            
        # except (OSError, IOError) as e:
        #     files = self.get_yaml_files(datasets_dir[dataset])
        #     json_data = self.json_data(metrics_type, files)
        #     self.raw_df = self.to_df(json_data)
        #     pickle.dump(self.raw_df, open('../temp_data/{}_{}_raw_df'.format(metrics_type, dataset), 'wb'))

        # try:
        #     self.df = pickle.load(open('../temp_data/{}_{}_df'.format(metrics_type, dataset), 'rb'))
        
        # except (OSError, IOError) as e:
        #     self.df = self.cleaning(self.raw_df)
        #     pickle.dump(self.df, open('../temp_data/{}_{}_df'.format(metrics_type, dataset), 'wb'))

        # self.typefilterpercentage = None

        # if file_type != 'all':
        #     self.typefilterpercentage = self.filter_filetype(file_type)

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



    def calculate_vectors(self, instanceblock): 
        vectorizer = CountVectorizer(token_pattern='[^\s]+').fit(instanceblock)
        vectorizer = vectorizer.transform(instanceblock)
        vectors =vectorizer.toarray()
        return vectors



    def calculate_cosine(self, vec1, vec2):
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]   



    def check_similarity(self, file_list):
        string_list = []
        for filePath in file_list:
            with open(filePath, 'r') as file:
                yml = file.read()
            string_list.append(yml)
        
        vectors = self.calculate_vectors(string_list)


        sims = []
        #todo: identifier for the similarities.

        for i in list(enumerate(vectors)):
            next_index = i[0] + 1

            for j in list(enumerate(vectors))[next_index:]:
                sims.append((i[0], j[0], self.calculate_cosine(i[1], j[1])))
        
        return sims         



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
        #Check similarity
        similarity_scores = self.check_similarity(list(df.index))
        similar_files = [pair for pair in similarity_scores if pair[2] == 1]

        #Because in order so multiple duplicates will be deleted eventually
        #here range because we have a numerical index, in the next one we have the actual index
        to_exclude = [pair[1] for pair in similar_files]
        ixs_to_keep = [ix for ix in range(df.shape[0]) if ix not in to_exclude]

        df = df.iloc[ixs_to_keep]

        #Check valid tosca
        tosca_metrics = ['na_count', 'nc_count', 'nc_min', 'nc_max', 'nc_median', 'nc_mean', 'ni_count',
       'nif_count', 'ninp_count', 'ninpc_count', 'nn_count', 'nnt_count', 'nout_count', 'np_count', 
       'np_min', 'np_max', 'np_median', 'np_mean', 'nr_count', 'nrt_count', 'ttb_check', 'cdnt_count', 
       'cdrt_count', 'cdat_count', 'cdct_count', 'cddt_count', 'cdgt_count', 'cdit_count', 'cdpt_count', 
       'nw_count', 'tdb_check', 'nrq_count', 'nsh_count', 'ncys_count', 'tob_check', 'ngc_count', 
       'ngp_count', 'ngro_count', 'npol_count', 'nf_count']

        check_tosca_df = df[tosca_metrics]
        check_tosca_df['valid_file'] = check_tosca_df.any(1)
        to_exclude = list(check_tosca_df[check_tosca_df['valid_file'] == False].index)
        ixs_to_keep = [ix for ix in list(df.index) if ix not in to_exclude]
        
        df = df.loc[ixs_to_keep]


        #Drop NaN rows and error columns, and make numeric
        df = df.drop(labels=(df.filter(regex='msg').columns), axis=1)
        df = df.dropna()
        cols = df.select_dtypes(include=['bool', 'object']).columns
        df[cols] = df[cols].astype(int)
        df = df.dropna()
        return df

    
    def filter_filetype(self, original_df):
        '''Filter on the file type. A file could be a service template, or containing
        custom type definitions, both or none of these two. It returns a dictionary with the indices
        belonging to each purpose.'''

        df = original_df.copy()

        custom_indicators = ['cdnt_count', 'cdrt_count', 'cdat_count', 'cdct_count', 'cddt_count', 'cdgt_count', 'cdit_count', 'cdpt_count']

        cus_df = df[custom_indicators]
        non_df = cus_df[(cus_df == 0).all(1)] 

        df['custom_def'] = [False if x in non_df.index else True for x in df.index]

        split_paths = {}
        split_paths['topology'] = df[(df['ttb_check'] == 1) & (df['custom_def'] == False)].index
        split_paths['custom'] = df[(df['ttb_check'] == 0) & (df['custom_def'] == True)].index
        split_paths['both'] = df[(df['ttb_check'] == 1) & (df['custom_def'] == True)].index

        assigned_indices = list(split_paths['topology']) + list(split_paths['custom']) + list(split_paths['both'])
        not_assigned_indices = [ix for ix in list(df.index) if ix not in assigned_indices]
        split_paths['none'] = df.loc[not_assigned_indices].index


        return split_paths


goal = 'all'
data = Data(goal)


# # %%
# def cleaning_test(df):
#         print('Start size: ', df.shape)
#         #Check similarity
#         similarity_scores = pickle.load(open('../temp_data/similarity_scores', 'rb'))
#         similar_files = [pair for pair in similarity_scores if pair[2] == 1]

#         #Because in order so multiple duplicates will be deleted eventually
#         #here range because we have a numerical index, in the next one we have the actual index
#         to_exclude = {pair[1] for pair in similar_files}
#         print('SIms to delete: ', len(to_exclude))
#         ixs_to_keep = [ix for ix in range(df.shape[0]) if ix not in to_exclude]
#         df = df.iloc[ixs_to_keep]

#         print('After sim deletion: ', df.shape)

#         #Check valid tosca
#         tosca_metrics = ['na_count', 'nc_count', 'nc_min', 'nc_max', 'nc_median', 'nc_mean', 'ni_count',
#        'nif_count', 'ninp_count', 'ninpc_count', 'nn_count', 'nnt_count', 'nout_count', 'np_count', 
#        'np_min', 'np_max', 'np_median', 'np_mean', 'nr_count', 'nrt_count', 'ttb_check', 'cdnt_count', 
#        'cdrt_count', 'cdat_count', 'cdct_count', 'cddt_count', 'cdgt_count', 'cdit_count', 'cdpt_count', 
#        'nw_count', 'tdb_check', 'nrq_count', 'nsh_count', 'ncys_count', 'tob_check', 'ngc_count', 
#        'ngp_count', 'ngro_count', 'npol_count', 'nf_count']

#         check_tosca_df = df[tosca_metrics]
#         check_tosca_df['valid_file'] = check_tosca_df.any(1)
#         to_exclude = list(check_tosca_df[check_tosca_df['valid_file'] == False].index)
#         print('Invalids to delete: ', len(to_exclude))
#         ixs_to_keep = [ix for ix in list(df.index) if ix not in to_exclude]
        
#         df = df.loc[ixs_to_keep]
#         print('After invalides deletion: ', df.shape)


#         #Drop NaN rows and error columns, and make numeric
#         df = df.drop(labels=(df.filter(regex='msg').columns), axis=1)
#         df = df.dropna()
#         cols = df.select_dtypes(include=['bool', 'object']).columns
#         df[cols] = df[cols].astype(int)
#         df = df.dropna()
#         print('After rest shit deletion: ', df.shape)
#         return df

# test_data = Data('all').raw_df
# x = cleaning_test(test_data)