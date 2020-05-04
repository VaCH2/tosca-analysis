import json
import os
from io import StringIO
from smells.metrics.cpl import CPL


def evaluate_script_with_rule(filePath):
    # RULE_FILENAME = os.path.join(os.path.dirname(__file__), 'rule_stats.json')

    # with open(RULE_FILENAME) as f:
    #     rule_dict = json.load(f)

    # q1 = rule_dict['cpl']['q1']
    # q3 = rule_dict['cpl']['q3']
    # iqr = q3 - q1

    with open(filePath, 'r', encoding='utf8') as f:
        yml = f.read()
        print(yml)
    yml = StringIO(yml.expandtabs(2))
    cpl_list = CPL(yml).count()
    print(cpl_list)

    smell_counter = 0

    for element in cpl_list:
        #if (element > q3 + 1.5*iqr) or (element < q1 - 1.5*iqr):
        if element > 140:
            smell_counter += 1

    return smell_counter

# path = r'dataminer\tmp\ystia\forge\Industry\apply_hostspool.yml'
# evaluate_script_with_rule(path)
    


    
