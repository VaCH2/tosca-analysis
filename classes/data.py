import os
import pandas as pd
from toscametrics import calculator
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

root_folder = os.path.dirname(os.path.dirname( __file__ ))
temp_data_folder = os.path.join(root_folder, 'temp_data')

class Data():
    #desnoods die metrics type hierin code ipv in die calculator en dan gewoon die hardcoden op 'all'
    def __init__(self, split='all', metrics_type='tosca_and_general'):
        '''A dictionary were the keys are the possible alternatives in the provided split.
        The value is the corresponding, filtered dataframe.'''

        try:
            raw_df = pickle.load(open(os.path.join(temp_data_folder, 'all_raw_df'), 'rb'))
            
        except (OSError, IOError):
            files = self.get_indices('all', None)
            json_data = self.json_data(metrics_type, files.get('all'))
            raw_df = self.to_df(json_data)
            pickle.dump(raw_df, open(os.path.join(temp_data_folder, 'all_raw_df'), 'wb'))


        self.raw_df = raw_df
        raw_size = self.raw_df.shape[0]


        try:
            df = pickle.load(open(os.path.join(temp_data_folder, 'all_df'), 'rb'))

        except (OSError, IOError):
            df = self.cleaning(self.raw_df)
            pickle.dump(df, open(os.path.join(temp_data_folder, 'all_df'), 'wb'))


        cleaned_size = df.shape[0]
        self.droppedrows = raw_size - cleaned_size
        split_indices = self.get_indices(split, df)
        #Include only valid 
        #Because get_indices looks at all files, so does not exclude the ones dropped during cleaning
        #We also rename the index to the relative path.
        filtered_dfs = {}
        
        for split, files in split_indices.items():
            files = [file.replace('c', 'C', 1) if file[0] == 'c' else file for file in files]
            files = [file for file in files if file in list(df.index)]
            
            if len(files) == 0:
                continue
            ix_mapping = {file : file.split('tmp\\')[1] for file in files}

            filtered_df = df.loc[files]
            filtered_df = filtered_df.rename(index=ix_mapping)
            filtered_dfs[split] = filtered_df

        self.dfs = filtered_dfs
        
        

    def get_indices(self, split, df):
        '''Filters the provided dataframe on the desired split and returns the indices of the filtered dataframe'''

        data_path = os.path.join(root_folder, 'dataminer', 'tmp')

        owners = [ item for item in os.listdir(data_path)]

        owner_and_repo = []
        for owner in owners:
            repos = os.listdir(os.path.join(data_path, owner))
            
            for repo in repos:
                owner_and_repo.append((owner, repo))
        
        professionalities = ['Example', 'Industry']

        if split == 'repo':
            split_paths = {f'{oar[0]}-{oar[1]}' : [os.path.join(data_path, oar[0], oar[1])] for oar in owner_and_repo}

        elif split == 'professionality':
            repo_paths = [os.path.join(data_path, oar[0], oar[1]) for oar in owner_and_repo]
            split_paths = {}
            
            for prof in professionalities:
                
                split_paths[prof] = [os.path.join(repo_path, prof) for repo_path in repo_paths]


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

                split_files[split] = files

        return split_files


    def get_yaml_files(self, path):
        '''Returns all the files with a YAML extension found in the provided path'''

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
        '''Transforms the provided instanceblock (string) into vectors'''

        vectorizer = CountVectorizer(token_pattern='[^\s]+').fit(instanceblock)
        vectorizer = vectorizer.transform(instanceblock)
        vectors =vectorizer.toarray()
        return vectors



    def calculate_cosine(self, vec1, vec2):
        '''Calculates the cosine similarity score'''

        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]   



    def check_similarity(self, file_list):
        '''Calculates the similarity score for each pair of the provided files and returns this in a list'''


        try:
            sims = pickle.load(open(os.path.join(temp_data_folder, 'similarity_scores'), 'rb'))

        except (OSError, IOError):
            string_list = []
            for filePath in file_list:
                try:
                    with open(filePath, 'r') as file:
                        yml = file.read()
                except UnicodeDecodeError:
                    with open(filePath, 'r', encoding='utf-8') as file:
                        yml = file.read()
                string_list.append(yml)
            
            vectors = self.calculate_vectors(string_list)


            sims = []

            for i in list(enumerate(vectors)):
                next_index = i[0] + 1

                for j in list(enumerate(vectors))[next_index:]:
                    sims.append((i[0], j[0], self.calculate_cosine(i[1], j[1])))
            
            pickle.dump(sims, open(os.path.join(temp_data_folder, 'similarity_scores'), 'wb'))
        
        return sims         



    def json_data(self, metrics_type, yaml_files):
        '''Calculates all the metrics over the provided files'''

        metrics = calculator.MetricCalculator(yaml_files, metrics_type).getresults
        return metrics



    def to_df(self, json_data):
        '''Transforms a JSON file to Pandas DataFrame'''

        flat_dict = {}
        for key, value in json_data.items():
            df = pd.io.json.json_normalize(value, sep='_')
            value = df.to_dict(orient='records')[0]
            flat_dict[key] = value

        df = pd.DataFrame.from_dict(flat_dict, orient='index')
        return df


    def cleaning(self, df):
        '''Applies cleaning steps on the provided dataframe. Steps are: delete similar files, 
        check if files are valid tosca files, drop error message columns, drop rows containing nan
        and make every column numeric.'''

        print('size raw df: ', df.shape)
        #Check similarity
        similarity_scores = self.check_similarity(list(df.index))
        similar_files = [pair for pair in similarity_scores if pair[2] == 1]

        #Because in order so multiple duplicates will be deleted eventually
        #here range because we have a numerical index, in the next one we have the actual index
        to_exclude = [pair[1] for pair in similar_files]
        ixs_to_keep = [ix for ix in range(df.shape[0]) if ix not in to_exclude]

        df = df.iloc[ixs_to_keep]
        print('size df after similarity deletion: ', df.shape)

        #Check valid tosca
        tosca_metrics = ['na_count', 'nc_count', 'nc_min', 'nc_max', 'nc_median', 'nc_mean', 'ni_count',
       'nif_count', 'ninp_count', 'ninpc_count', 'nn_count', 'nout_count', 'np_count', 
       'np_min', 'np_max', 'np_median', 'np_mean', 'nr_count', 'ttb_check', 'cdnt_count', 
       'cdrt_count', 'cdat_count', 'cdct_count', 'cddt_count', 'cdgt_count', 'cdit_count', 'cdpt_count', 
       'nw_count', 'tdb_check', 'nrq_count', 'nsh_count', 'ncys_count', 'tob_check',
       'ngro_count', 'npol_count', 'nf_count']

        check_tosca_df = df[tosca_metrics]
        check_tosca_df['valid_file'] = check_tosca_df.any(1)
        to_exclude = list(check_tosca_df[check_tosca_df['valid_file'] == False].index)
        ixs_to_keep = [ix for ix in list(df.index) if ix not in to_exclude]
        
        df = df.loc[ixs_to_keep]
        print('size df after invalid TOSCA deletion: ', df.shape)

        #Drop NaN rows and error columns, and make numeric
        df = df.drop(labels=(df.filter(regex='msg').columns), axis=1)
        df = df.dropna()
        cols = df.select_dtypes(include=['bool', 'object']).columns
        df[cols] = df[cols].astype(int)
        df = df.dropna()

        print('size df after NaN and error column drops: ', df.shape)
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