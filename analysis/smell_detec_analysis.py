#Voorbeeld hoe de nieuwe module te importeren en hoe je ze vervoglens kan gebruiken om files er doorheen te trekken.
#Met de main kun je de rules opnieuw uitrekenen, maar is eigenlijk niet meer nodig tenzij er wat wordt aangepast. 

import sys
sys.path.append('C:/Users/s145559/OneDrive - TU Eindhoven/School/JADS/Jaar 2/Thesis/RADON PROJECT/GIT projects/TOSCASmellDetector')
from smelldetector.longstatement import evaluate_script_with_rule
from smellmetrics.cpl import CPL
from smelldetector.rule_calculator import main

path = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\GIT projects\tosca-metrics\test_files\cpl.yaml'
evaluate_script_with_rule(path)