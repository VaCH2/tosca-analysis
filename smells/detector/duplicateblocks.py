import os
from io import StringIO
from smells.metrics.blueprint_metric import BlueprintMetric
from smells.utils import keyValueList
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from smells.utils import calculate_depth

# def evaluate_script_with_rule(filePath):

#     if isinstance(filePath, StringIO):
#         strio = filePath.getvalue()
    
#     else:
#         with open(filePath, 'r', encoding='utf8') as f:
#             raw_yml = f.read()
#         raw_yml = StringIO(raw_yml.expandtabs(2))
#         strio = BlueprintMetric(raw_yml).getStringIOobject

#     #Update these with all the blocks I want to check on duplicates
#     goal_dicts = {'node_types:' : [],
#               'relationship_types:' : [],
#               'data_types:' : [], 
#               'capability_types:' : [],
#               'artifact_types:' : [],
#               'node_templates:' : [],
#               'requirements:' : [],
#               'interfaces:' : [],
#               'properties:' : []          
#     }

#     for k, v in goal_dicts.items():
#         blocks = get_block(k, strio)
#         v.extend(blocks)
    
#     return goal_dicts

#         try:
#             vectors = calculate_vectors(v)

#             sims = []
#             #todo: identifier for the similarities.

#             for i in list(enumerate(vectors)):
#                 next_index = i[0] + 1

#                 for j in list(enumerate(vectors))[next_index:]:
#                     sims.append(calculate_similarity(i[1], j[1]))

#             goal_dicts[k] = sims

#         except ValueError:
#             pass
 
#     return goal_dicts

# def get_block(key, script):
#     '''Creates blocks of code which have a similar goal based on the key provided'''

#     split_lines = script.splitlines()
#     split_lines = calculate_depth(split_lines)

#     start_keyblock = False
#     start_instanceblock = False
#     keyblock_end = -1
#     instanceblock_end = -1

#     blocks = []
#     block = []
#     last_line = len(split_lines)
#     count = 0

#     for line in split_lines:
#         count += 1

#         if line[1] <= keyblock_end and len(line[0].strip()) != 0:
#             start_keyblock = False

#         if line[1] <= instanceblock_end and start_instanceblock == True:
#             start_instanceblock = False
#             if len(block) != 0:
#                 cleaned_block = clean_instanceblock(block)
#                 blocks.extend(cleaned_block)
#             block = [] 

#         if start_keyblock:
#             if start_instanceblock == False:
#                 instanceblock_end = line[1]
#             start_instanceblock = True        
        
#         if start_instanceblock and len(line[0].strip()) != 0:
#             block.append(line[0])      

#         if key in line[0]:
#             start_keyblock = True
#             keyblock_end = line[1]

#         if count == last_line and len(block) != 0:
#             cleaned_block = clean_instanceblock(block)
#             blocks.extend(cleaned_block)

#     return blocks


# def clean_instanceblock(instanceblock):
#     instanceblock = [str(word).lower() for word in instanceblock]
#     instanceblock = [' '.join(word for word in instanceblock)]
#     instanceblock = [word.strip() for word in instanceblock]

#     return instanceblock


# def calculate_vectors(instanceblock): 
#     vectorizer = CountVectorizer(token_pattern='[^\s]+').fit(instanceblock)
#     vectorizer = vectorizer.transform(instanceblock)
#     vectors =vectorizer.toarray()

#     return vectors


# def calculate_similarity(vec1, vec2):
#     vec1 = vec1.reshape(1, -1)
#     vec2 = vec2.reshape(1, -1)

#     return cosine_similarity(vec1, vec2)[0][0]



def evaluate_script_with_rule(filePath):

    if isinstance(filePath, StringIO):
        strio = filePath.getvalue()
    
    else:
        with open(filePath, 'r', encoding='utf8') as f:
            raw_yml = f.read()
        raw_yml = StringIO(raw_yml.expandtabs(2))
        strio = BlueprintMetric(raw_yml).getStringIOobject

    strio_length = len(strio)
    code_chunks = [strio[i:150+i] for i in range(0, strio_length - 149)]

    duplicate_blocks = [chunk for chunk in code_chunks if strio.count(chunk) > 1]

    return len(duplicate_blocks)




#path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\All\types.yaml'
#path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\All\tosca-node-type (2).yml'

#path = 'db_test.yaml'
#results = evaluate_script_with_rule(path)
