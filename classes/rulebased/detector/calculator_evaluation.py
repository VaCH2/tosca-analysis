from smells.detector import longstatement
from smells.detector import toomanyattributes
from smells.detector.rule_calculator import main
from smells.utils import get_yaml_files
import numpy as np
import random

path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\All'

files = get_yaml_files(path)
random.shuffle(files)
split = int(0.75*len(files))

train_files = files[:split]
test_files = files[split:]

#Calculate new rules based on the trainset
main(train_files)

for script in test_files:
    print(script.split('\\')[-1], '- ', 
    'Long Statement :', longstatement.evaluate_script_with_rule(script), '\n',
    'Too many Attributes :', toomanyattributes.evaluate_script_with_rule(script)
    )

#What are we doiing with the test split?

