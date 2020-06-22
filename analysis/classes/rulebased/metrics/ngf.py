from io import StringIO
from classes.rulebased.metrics.blueprint_metric import BlueprintMetric
from toscametrics.metrics.ninp import NINP
from toscametrics.metrics.nout import NOUT

class NGF(BlueprintMetric):
    """ This class is responsible for providing the methods to count the number of get functions (NGF) in a given .yaml file."""
    
    def count(self):
        '''Returns the number of get_attribute, get_property, get_operation_output, and get_input instances '''
        
        inputs = NINP(StringIO(self.getStringIOobject.expandtabs(2))).count()
        outputs = NOUT(StringIO(self.getStringIOobject.expandtabs(2))).count()
         
        raw_string = self.getStringIOobject
        get_attribute_count = raw_string.count('get_attribute')
        get_property_count = raw_string.count('get_property')
        get_operation_output_count = raw_string.count('get_operation_output')
        get_input_count = raw_string.count('get_input')

        return inputs + outputs + get_attribute_count + get_property_count + get_operation_output_count + get_input_count