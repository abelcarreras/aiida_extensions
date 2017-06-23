from aiida import load_dbenv
load_dbenv()

from aiida.orm import load_node, load_workflow
from aiida.orm import Code, DataFactory

import matplotlib.pyplot as plt

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
KpointsData = DataFactory('array.kpoints')

import numpy as np

#######################
wf = load_workflow(1086)
#######################

thermal_properties = wf.get_result('thermal_properties')

energy = thermal_properties.get_array('electronic_energies')
volumes = thermal_properties.get_array('volumes')
entropy = thermal_properties.get_array('entropy')
cv = thermal_properties.get_array('cv')
temperature = thermal_properties.get_array('temperature')

plt.figure(1)

plt.plot(volumes, energy)

plt.figure(2)

for i, w in enumerate(wf.get_steps()[1].get_sub_workflows()):
    frequencies = [w.get_result('quasiparticle_data').get_dict()['{}'.format(k)]['q_point_0']['4']['frequency'] for k in range(100,800,100)]
    plt.plot(frequencies)

plt.show()

