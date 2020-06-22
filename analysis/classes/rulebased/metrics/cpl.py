from io import StringIO
from classes.rulebased.metrics.blueprint_metric import BlueprintMetric

class CPL(BlueprintMetric):
    """ This class is responsible for providing the methods to count the characters per line (CPL) in a given .yaml file."""
    
    def count(self):
        '''Creates a list of character counts per line bigger than 5 and are not comments'''
        list_of_character_counts = []

        for l in self.getStringIOobject.splitlines():
            #l = str(l.strip())
            l = l.split('#')[0]
            if len(l)>5 and len(l.strip()) != 0:
                list_of_character_counts.append(len(l))


        return list_of_character_counts