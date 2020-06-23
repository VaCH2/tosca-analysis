import json
import os
from io import StringIO
from rulebased.metrics.cpl import CPL


def evaluate_script_with_rule(filePath):

    try:
        with open(filePath, 'r', encoding='utf8') as f:
            yml = f.read()
        yml = StringIO(yml.expandtabs(2))
        cpl_list = CPL(yml).count()
        smell_counter = 0

        for element in cpl_list:
            if element > 140:
                smell_counter += 1

        return smell_counter


    except Exception as e:
        return e
    


    
