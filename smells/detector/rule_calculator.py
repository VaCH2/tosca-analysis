from smells.utils import get_yaml_files
from smells.metrics.import_metrics import metrics
import os
from io import StringIO
import json
import numpy as np

#Depreciated, we don't use rules anymore. We just do what Schwarz and Sharma did. 

def main(custom_set_of_scripts=None):
    path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\All\All'

    if isinstance(custom_set_of_scripts, list):
        files = custom_set_of_scripts
    else:
        files = get_yaml_files(path)
    
    calc_metrics = calculate_metrics(files)
    rule_stats = bootstrap_metrics(calc_metrics)

    RULE_FILENAME = os.path.join(os.path.dirname(__file__), 'rule_stats.json')

    with open(RULE_FILENAME, 'w') as fp:
            json.dump(rule_stats, fp)


def calculate_metrics(allFiles):
    calculated_metrics = {}

    for metric_name, metric_class in metrics.items():
        calculated_metrics[metric_name] = []

        for filePath in allFiles:
            with open(filePath, 'r', encoding='utf8') as file:
                yml = file.read()
            yml = StringIO(yml.expandtabs(2))

            value = metric_class(yml).count()
            if isinstance(value, list):
                calculated_metrics[metric_name].extend(value)
            else:
                calculated_metrics[metric_name].append(value)

    return calculated_metrics


def bootstrap_metrics(calc_metrics):
    metric_stats = {}

    for metric_name, metric_values in calc_metrics.items():
        metric_stats[metric_name] = {}

        q1, median, q3 = bootstrap_simulation(metric_values)
        metric_stats[metric_name]['q1'] = q1
        metric_stats[metric_name]['median'] = median
        metric_stats[metric_name]['q3'] = q3

    return metric_stats


def bootstrap_simulation(data, n_sample=100, repetitions=100):
    boot_Q1s = []
    boot_medians = []
    boot_Q3s = []

    for _ in range(repetitions):
        bootsample = np.random.choice(data, size=n_sample, replace=True)

        boot_Q1s.append(np.percentile(bootsample, 25))
        boot_medians.append(np.median(bootsample))
        boot_Q3s.append(np.percentile(bootsample, 75))

    q1 = np.median(boot_Q1s)
    median = np.median(boot_medians)
    q3 = np.median(boot_Q3s)
    
    return  q1, median, q3


if __name__ == "__main__":
    main()
