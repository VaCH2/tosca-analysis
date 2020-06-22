from classes.rulebased.detector import longstatement
from classes.rulebased.detector import toomanyattributes
from classes.rulebased.detector import duplicateblocks
from classes.rulebased.detector import longresource
from classes.rulebased.detector import insufficientmodularization
from classes.rulebased.detector import weakenedmodularity

from classes.rulebased.detector.rule_calculator import main
from classes.rulebased.utils import get_yaml_files
import numpy as np
import pandas as pd
import os
import pickle
from classes.data import Data

root_folder = os.path.dirname(os.path.dirname( __file__ ))
temp_data_folder = os.path.join(root_folder, 'temp_data')
data_path = os.path.join(root_folder, 'dataminer', 'tmp')

try:
    smells_df = pickle.load(open(os.path.join(temp_data_folder, 'smells_df'), 'rb'))

except (OSError, IOError):

    files = get_yaml_files(data_path)

    results = {}

    for script in files:
        results[script.split('tmp\\')[1]] = {
            'ls' : longstatement.evaluate_script_with_rule(script), 
            'tma' : toomanyattributes.evaluate_script_with_rule(script),
            'db' : duplicateblocks.evaluate_script_with_rule(script),
            'lr' : longresource.evaluate_script_with_rule(script),
            'im' : insufficientmodularization.evaluate_script_with_rule(script),
            'wm' : weakenedmodularity.evaluate_script_with_rule(script)
        }

    smells_df = pd.DataFrame(results).T

    sound_ixs = Data().dfs.get('all').index.values
    smells_df = smells_df[smells_df.index.isin(sound_ixs)]
    smells_df = smells_df.drop(r'SeaCloudsEU\tosca-parser\Industry\normative_types.yaml')
    
    pickle.dump(smells_df, open(os.path.join(temp_data_folder, 'smells_df'), 'wb'))

