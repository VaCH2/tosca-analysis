import json
import os
from io import StringIO
from smells.metrics.blueprint_metric import BlueprintMetric
from toscametrics.yml.loc import LOC



def evaluate_script_with_rule(filePath):

    try:
        with open(filePath, 'r', encoding='utf8') as f:
            yml = f.read()
        yml = StringIO(yml.expandtabs(2))

        blueprint = BlueprintMetric(yml)
        imports = blueprint.getyml.get('imports')
        loc_count = LOC(yml).count()
        smell_detected = False

        if isinstance(imports, list):
            interproject_imports = []
            intraproject_imports = []
            for imp in imports:
                if 'http' in imp or 'snapshot' in imp:
                    interproject_imports.append(imp)
                else:
                    interproject_imports.append(imp)

            
            try:
                if len(interproject_imports) / loc_count > 0.1:
                    smell_detected = True
            except:
                pass
            
        return smell_detected

    except Exception as e:
        return e




# path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\GIT projects\ANALYSIS\dataminer\tmp\SeaCloudsEU\SeaCloudsPlatform\Industry\webchat_adp-iaas.yml'
# result = evaluate_script_with_rule(path)