from io import StringIO
from smells.metrics.blueprint_metric import BlueprintMetric

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





# string = 'tosca_definitions_version: tosca_simple_yaml_1_3tosca_simple_yaml_1_3\n\ndescription: Template for deploying a single server with predefined properties.\n\ntopology_template: #This is the topology template\n  node_templates:\n    db_server:\n      type: tosca.nodes.Compute\n      # Omitted'
# yml = StringIO(string.expandtabs(2)) 
# metric = CPL(yml)

# print(string)
# print('CPL count: ', metric.count())