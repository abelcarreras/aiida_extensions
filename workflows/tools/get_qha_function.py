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

import matplotlib.pyplot as plt

def calculate_qha_inline(**kwargs):

    from phonopy import PhonopyQHA
    import numpy as np


#    thermal_properties_list = [key for key, value in kwargs.items() if 'thermal_properties' in key.lower()]
#    optimized_structure_data_list = [key for key, value in kwargs.items() if 'optimized_structure_data' in key.lower()]
    structure_list = [key for key, value in kwargs.items() if 'final_structure' in key.lower()]

    volumes = []
    electronic_energies = []
    fe_phonon = []
    entropy = []
    cv = []

    for i in range(len(structure_list)):
       # volumes.append(kwargs.pop(key).get_cell_volume())
       volumes.append(kwargs.pop('final_structure_{}'.format(i)).get_cell_volume())
       electronic_energies.append(kwargs.pop('optimized_structure_data_{}'.format(i)).dict.energy)
       thermal_properties = kwargs.pop('thermal_properties_{}'.format(i))
       temperatures = thermal_properties.get_array('temperature')
       fe_phonon.append(thermal_properties.get_array('free_energy'))
       entropy.append(thermal_properties.get_array('entropy'))
       cv.append(thermal_properties.get_array('cv'))

    sort_index = np.argsort(volumes)

    temperatures = np.array(temperatures)
    volumes = np.array(volumes)[sort_index]
    electronic_energies = np.array(electronic_energies)[sort_index]
    fe_phonon = np.array(fe_phonon).T[:, sort_index]
    entropy = np.array(entropy).T[:, sort_index]
    cv = np.array(cv).T[:, sort_index]





    # Calculate QHA
    phonopy_qha = PhonopyQHA(volumes.tolist(),
                             electronic_energies.tolist(),
                             eos="vinet",
                             temperatures=temperatures.tolist(),
                             free_energy=fe_phonon.tolist(),
                             cv=cv.tolist(),
                             entropy=entropy.tolist(),
                             #                         t_max=options.t_max,
                             verbose=True)


    qha_output = ArrayData()
    qha_output.set_array('temperature', np.array([1, 2, 3, 4]))
    qha_output.store()
    print qha_output
    exit()


    # Get data
    qha_temperatures = phonopy_qha._qha._temperatures[:phonopy_qha._qha._max_t_index]
    helmholtz_volume = phonopy_qha.get_helmholtz_volume()
    thermal_expansion = phonopy_qha.get_thermal_expansion()
    volume_temperature = phonopy_qha.get_volume_temperature()
    heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
    volume_expansion = phonopy_qha.get_volume_expansion()
    gibbs_temperature = phonopy_qha.get_gibbs_temperature()


    qha_output = ArrayData()

    qha_output.set_array('temperature', np.array(qha_temperatures))
    qha_output.set_array('helmholtz_volume', np.array(helmholtz_volume))
    qha_output.set_array('thermal_expansion', np.array(thermal_expansion))
    qha_output.set_array('volume_temperature', np.array(volume_temperature))
    qha_output.set_array('heat_capacity_P_numerical', np.array(heat_capacity_P_numerical))
    qha_output.set_array('volume_expansion', np.array(volume_expansion))
    qha_output.set_array('gibbs_temperature', np.array(gibbs_temperature))

    qha_output.store()

    return {'qha_data': qha_output}




#######################
wf = load_workflow(910)
#######################



wf_parameters = wf.get_parameters()

# self.get_step_calculations(self.optimize).latest('id')

interval = wf.get_attribute('interval')

max = wf.get_attribute('max')
min = wf.get_attribute('min')

n_points = int((max - min) / interval) + 1
test_pressures = [min + interval * i for i in range(n_points)]


# Remove duplicates
wf_complete_list = list(wf.get_step('pressure_expansions').get_sub_workflows())
wf_complete_list += list(wf.get_step('collect_data').get_sub_workflows())
wf_complete_list += list(wf.get_step('complete').get_sub_workflows())


volumes = []
electronic_energies = []
temperatures = []
fe_phonon = []
entropy = []
cv = []


inline_params = {}

for wf_test in wf_complete_list:
    for i, pressure in enumerate(test_pressures):
        if wf_test.get_attribute('pressure') == pressure:
            thermal_properties = wf_test.get_result('thermal_properties')
            optimized_data = wf_test.get_result('optimized_structure_data')
            final_structure = wf_test.get_result('final_structure')

            inline_params.update({'thermal_properties_{}'.format(i): thermal_properties})
            inline_params.update({'optimized_structure_data_{}'.format(i): optimized_data})
            inline_params.update({'final_structure_{}'.format(i): final_structure})


qha_result = calculate_qha_inline(**inline_params)

print qha_result['qha_data']
