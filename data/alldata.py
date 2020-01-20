from data import Data

class Alldata(Data):
    def __init__(self, path):
        x = self.get_yaml_files(path)
        print(len(x))


test = r'C:\Users\s145559\OneDrive - TU Eindhoven\School\JADS\Jaar 2\Thesis\RADON PROJECT\Data\\1. Total Examples'
hup = Alldata(test)