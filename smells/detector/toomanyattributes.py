import json
import os
from io import StringIO
from smells.metrics.ngf import NGF
from toscametrics.yml.loc import LOC

def evaluate_script_with_rule(filePath):
    # RULE_FILENAME = os.path.join(os.path.dirname(__file__), 'rule_stats.json')

    # with open(RULE_FILENAME) as f:
    #     rule_dict = json.load(f)

    # q1 = rule_dict['ngf']['q1']
    # q3 = rule_dict['ngf']['q3']
    # iqr = q3 - q1

    with open(filePath, 'r', encoding='utf8') as f:
        yml = f.read()
    yml = StringIO(yml.expandtabs(2))
    attributes = NGF(yml).count()
    loc_count = LOC(yml).count()
    
    smell_detected = False
    try:
        if attributes / loc_count > 0.5:
            smell_detected = True
    except:
        pass
       

    return smell_detected

# path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\All\alien4cloud-config (3).yml'
# evaluate_script_with_rule(path)
    


    
