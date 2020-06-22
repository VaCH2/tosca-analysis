from io import StringIO
from smells.metrics.blueprint_metric import BlueprintMetric
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




# string = 'tosca_definitions_version: alien_dsl_2_0_0\n\nmetadata:\n  template_name: org.ystia.dns.resolvconf.ansible\n  template_version: 2.2.0-SNAPSHOT\n  template_author: Ystia\n\n\nimports:\n  - tosca-normative-types:1.0.0-ALIEN20\n  - yorc-types:1.1.0\n  - org.ystia.dns.pub:2.2.0-SNAPSHOT\n\nnode_types:\n  org.ystia.dns.resolvconf.ansible.Resolvconf:\n    attributes:\n      hostname: { get_operation_output: [SELF, Standard, configure, HOSTNAME] }\n    requirements:\n      - dns_server: \n          capability: org.ystia.dns.pub.capabilities.DnsEndpoint\n          relationship: org.ystia.dns.resolvconf.ansible.relationships.ConnectsTo\n          occurrences: [1, UNBOUNDED] \n    interfaces:\n      Standard:\n        configure:\n          inputs:\n            DOMAIN: { get_property: [SELF, domain]}\n            SEARCH: { get_property: [SELF, search]}\n          implementation: playbooks/configure.yaml\n          \n    \nrelationship_types:\n  org.ystia.dns.resolvconf.ansible.relationships.ConnectsTo:\n    derived_from: tosca.relationships.ConnectsTo\n    interfaces:\n      Configure:\n        inputs:\n          IP_ADDRESS: {get_attribute: [TARGET, private_address]}\n        add_target: playbooks/add_target_dns_server.yaml\n        remove_target: playbooks/remove_target_dns_server.yaml\n        add_source:\n          inputs:\n            HOSTNAME: {get_attribute: [SOURCE, hostname]}\n            IP_ADDRESS: {get_attribute: [SOURCE, private_address]}\n          implementation: playbooks/add_host.yaml\n        remove_source:\n          inputs:\n            HOSTNAME: {get_attribute: [SOURCE, hostname]}\n            IP_ADDRESS: {get_attribute: [SOURCE, private_address]}\n          implementation: playbooks/remove_host.yaml\n          '
# yml = StringIO(string.expandtabs(2)) 
# metric = NGF(yml)

# print(string)
# print('NPF count: ', metric.count())