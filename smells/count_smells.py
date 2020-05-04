from smells.detector import longstatement
from smells.detector import toomanyattributes
from smells.detector.rule_calculator import main
from smells.utils import get_yaml_files
import numpy as np
import pandas as pd

path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\All'

files = get_yaml_files(path)

results = {}

for script in files:
    results[script.split('\\')[-1]] = {'ls' : longstatement.evaluate_script_with_rule(script), 
    'tma' : toomanyattributes.evaluate_script_with_rule(script)}

results = pd.DataFrame(results).T

#%%
results['ls'].value_counts()
# %%
results['tma'].value_counts()
