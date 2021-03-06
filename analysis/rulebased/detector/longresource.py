import json
import os
from io import StringIO
from rulebased.utils import calculate_depth


def evaluate_script_with_rule(filePath):

    try:
    
        with open(filePath, 'r', encoding='utf8') as f:
            yml = f.read()
        yml = StringIO(yml.expandtabs(2))

        depth_per_line = calculate_depth(yml)
        interfaces_loc = []
        start = False

        for line in depth_per_line:
            if 'interface' in line[0]:
                depth = line[1]
                lines = 0
                start = True
                continue

            if start == True:
                if line[1] > depth:
                    lines += 1
                else:
                    start = False
                    interfaces_loc.append(lines)

            


        return len([interface for interface in interfaces_loc if interface > 7])

    except Exception as e:
        return e