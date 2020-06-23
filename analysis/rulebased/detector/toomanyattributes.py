import json
import os
from io import StringIO
from rulebased.metrics.ngf import NGF
from toscametrics.yml.loc import LOC

def evaluate_script_with_rule(filePath):

    try:
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

    except Exception as e:
        return e
    


    
