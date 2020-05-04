import json
import os
from io import StringIO
from smells.utils import calculate_depth


def evaluate_script_with_rule(filePath):

    try:
        with open(filePath, 'r', encoding='utf8') as f:
            yml = f.read()
        yml = StringIO(yml.expandtabs(2))

        depth_per_line = calculate_depth(yml)

        # Situation 2: Too long topology template
        topology_template_loc = []
        start = False

        for line in depth_per_line:
            if 'topology_template' in line[0]:
                depth = line[1]
                lines = 0
                start = True
                continue

            if start == True:
                if line[1] > depth:
                    lines += 1
                else:
                    start = False
                    topology_template_loc.append(lines)

        topology_template_loc = [et for et in topology_template_loc if et > 40]


        #Situation 3: Too complex topology template
        complexity_elements = ['interfaces:', 'requirements:', 'node_filter:']

        complexity_depths = []
        start = False

        for line in depth_per_line:
            if line[0] in complexity_elements:
                depth = line[1]
                max_depth = []
                start = True
                continue

            if start == True:
                if line[1] >= depth:
                    max_depth.append(line[1] - depth)
                else:
                    start = False
                    if len(max_depth) != 0:
                        complexity_depths.append(max(max_depth))

        complexity_depths = [dp for dp in complexity_depths if dp > 3]

        return len(topology_template_loc) + len(complexity_depths)


    except Exception as e:
        return e

# path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\GIT projects\ANALYSIS\dataminer\tmp\SeaCloudsEU\SeaCloudsPlatform\Industry\webchat_adp-iaas.yml'
# result = evaluate_script_with_rule(path)