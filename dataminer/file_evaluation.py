import os
import shutil
from utils import all_filepaths
from utils import get_valid_files
from utils import remove_invalid_files
from utils import safe_copy
import yaml

root_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.join(root_folder, 'tmp')

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

example_indicators = ['test', 'tests', 'example', 'examples', 'hello-world', 'helloworld', 'tutorial']


owners = [ item for item in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, item)) ]

owner_and_repo = []
for owner in owners:
    repos = os.listdir(os.path.join(root_folder, owner))
    
    for repo in repos:
        owner_and_repo.append((owner, repo))


for repo in owner_and_repo:
    path = os.path.join(root_folder, repo[0], repo[1])

    example_dir = os.path.join(path, 'Example')
    industry_dir = os.path.join(path, 'Industry')
    os.makedirs(example_dir)
    os.makedirs(industry_dir)

    files = all_filepaths(path)
    files = get_valid_files(files)


    for file in files:
        
        if 'travis' in file:
            continue

        file_path = os.path.join(path, file.strip('\\'))
        
        if file_path.endswith('.tosca'):
            with open(file_path, 'r', encoding='utf8') as f:
                raw_text = f.read()

            if 'xml' not in raw_text:
                new_file = file_path.replace('.tosca', '.yaml', 1)
                os.rename(file_path, new_file)
                file_path = new_file
            
            else:
                continue


        if sum([file.count(indicator) for indicator in example_indicators]) != 0:
            safe_copy(file_path, example_dir)

        else:
            safe_copy(file_path, industry_dir)

    remove_invalid_files(path)
        



