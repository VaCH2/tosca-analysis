#Voorbeeld hoe de nieuwe module te importeren en hoe je ze vervoglens kan gebruiken om files er doorheen te trekken.
#Met de main kun je de rules opnieuw uitrekenen, maar is eigenlijk niet meer nodig tenzij er wat wordt aangepast. 

import sys
#sys.path.append('C:/Users/s145559/OneDrive - TU Eindhoven/School/JADS/Jaar 2/Thesis/RADON PROJECT/GIT projects/TOSCASmellDetector')
sys.path.append(r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\GIT projects\TOSCASmellDetector')
from smells.detector import longstatement
from smells.detector import toomanyattributes

from scipy.stats import mannwhitneyu
from data import Data

#nog fixen dat alle opties makkelijk gedaan kunnen worden
#Ff zorgen dat tussenresulaten opgeslagen worden

def main(set1, set2):
    sets = [set1, set2]
    file_ixs = get_data(sets)
    smells = get_smells(file_ixs)
    sig = get_sig(smells, set1, set2)
    return sig

def get_data(datasets):
    ix_lists = {}
    for dataset in datasets:
        data = Data('tosca_and_general', dataset)
        ix_list = [r'{}'.format(ix) for ix in data.df.index]
        ix_lists[dataset] = ix_list
    return ix_lists


def get_smells(ix_lists):
    smells = {}
    for dataset, ix_list in ix_lists.items():
        tma = [toomanyattributes.evaluate_script_with_rule(ix) for ix in ix_list]
        #ls = ...
        smells[dataset] = tma
    return smells


def get_sig(smell_data, set1, set2):
    stat, p = mannwhitneyu(smell_data[set1], smell_data[set2])
    return (stat, p)

main('industry', 'example')
