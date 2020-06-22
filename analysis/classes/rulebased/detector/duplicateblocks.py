import os
from io import StringIO
from classes.rulebased.metrics.blueprint_metric import BlueprintMetric
from classes.rulebased.utils import keyValueList
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from classes.rulebased.utils import calculate_depth


def evaluate_script_with_rule(filePath):

    try:
        if isinstance(filePath, StringIO):
            strio = filePath.getvalue()
        
        else:
            with open(filePath, 'r', encoding='utf8') as f:
                raw_yml = f.read()
            raw_yml = StringIO(raw_yml.expandtabs(2))
            strio = BlueprintMetric(raw_yml).getStringIOobject

        strio_length = len(strio)
        code_chunks = [strio[i:150+i] for i in range(0, strio_length - 149)]

        duplicate_blocks = [chunk for chunk in code_chunks if strio.count(chunk) > 1]

        return len(duplicate_blocks)

    except Exception as e:
        return e
