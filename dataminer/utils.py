import os
import shutil
import yaml
import errno
import stat
from datetime import datetime
import time

from dulwich import porcelain


def all_filepaths(path_to_root):
    """
    Get the set of all the files in a folder (excluding .git directory).
    
    Parameters
    ----------
    path_to_root : str : the path to the root of the directory to analyze.
    Return
    ----------
    filepaths : set : the set of strings (filepaths).
    """

    files = set()

    for root, _, filenames in os.walk(path_to_root):
        if '.git' in root:
            continue
        for filename in filenames: 
            path = os.path.join(root, filename)
            path = path.replace(path_to_root, '')
            if path.startswith('/'):
                path = path[1:]

            files.add(path)
    
    return files

def clone_repo(owner: str, name: str):
    """
    Clone a repository on local machine.
    
    Parameters
    ----------
    owner : str : the name of the owner of the repository.
    name : str : the name of the repository.
    """
    
    try:
        path_to_owner_name = os.path.join('tmp', owner, name)

        if not os.path.isdir(path_to_owner_name):
            
            os.makedirs(path_to_owner_name)
            clone_obj = porcelain.clone(f'https://github.com/{owner}/{name}.git', path_to_owner_name)
            clone_obj.close()

    except Exception as e:
        with open("logs/clone_errors.txt", "a+") as file:
            file.write(datetime.now() + ' :' + e + "\n")


def handleRemoveReadonly(func, path, exc):
    """
    Function which handles permission when files are not accessable in Windows.
    """
    excvalue = exc[1]

    if func in (os.rmdir, os.unlink, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) 
        func(path)


def delete_repo(owner: str, name: str):
    """
    Delete a local repository.
    
    Parameters
    ----------
    owner : str : the name of the owner of the repository.
    name : str : the name of the repository.
    """

    path_to_delete = os.path.join('tmp', owner)

    owners_repos = len([file for file in os.listdir(path_to_delete) if os.path.isdir(os.path.join(path_to_delete, file))])

    if owners_repos > 1:
        path_to_delete = os.path.join(path_to_delete, name)

    try:
        time.sleep(15)
        shutil.rmtree(path_to_delete, ignore_errors=False, onerror=handleRemoveReadonly)
    
    except:
        with open("logs/to_delete.txt", "a+") as file:
            file.write(path_to_delete + "\n")


        
def check_tosca(path):
    """
    Opens a YAML file and determines if it is a tosca_script.
    
    Parameters
    ----------
    path : str : the path to the YAML file to check.
    Return
    ----------
    tosca : boolean : a boolean if the script is tosca or not.
    """

    if 'travis' in path:
        return False

    try:
        with open(path, 'r', encoding='utf8') as f:
            yml = f.read()

        yml = yaml.safe_load(yml)
        return 'tosca_definitions_version' in yml.keys()
        
    except:
        return False



def get_valid_files(file_set):
    '''All files which have the .yaml, .yml or .tosca extension are returned'''
    
    valid_files = [ file for file in file_set if file.endswith( ('.yaml','.yml', '.tosca') ) ]
    return valid_files



def remove_invalid_files(path):
    '''Deletes all the files and directories in the provided path, excluding the Example and 
    Industry directory'''

    entities = os.listdir(path)
    excluded_dirs = ['Example', 'Industry']

    for entity in entities:
        if entity in excluded_dirs:
            continue

        elif os.path.isfile(os.path.join(path, entity)):
            os.remove(os.path.join(path, entity))

        elif os.path.isdir(os.path.join(path, entity)):
            shutil.rmtree(os.path.join(path, entity), ignore_errors=False, onerror=handleRemoveReadonly)