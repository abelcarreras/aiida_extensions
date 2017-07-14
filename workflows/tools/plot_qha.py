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


def get_phonon(structure, force_constants, phonopy_input):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'],
                     symprec=phonopy_input['symmetry_precision'])

    phonon.set_force_constants(force_constants)

    return phonon


def calculate_qha_inline(**kwargs):

    from phonopy import PhonopyQHA
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    import numpy as np

 #   structures = kwargs.pop('structures')
 #   optimized_data = kwargs.pop('optimized_data')
 #   thermodyamic_properties = kwargs.pop('thermodyamic_properties')

    entropy = []
    cv = []
  #  volumes = []
    fe_phonon = []
    temperatures = None

 #   structures = [key for key, value in kwargs.items() if 'structure' in key.lower()]
 #   for key in structures:
 #       volumes.append(kwargs.pop(key).get_cell_volume())

    volumes = [value.get_cell_volume() for key, value in kwargs.items() if 'structure' in key.lower()]
    electronic_energies = [value.get_dict()['energy'] for key, value in kwargs.items() if 'optimized_data' in key.lower()]

    thermal_properties_list = [key for key, value in kwargs.items() if 'thermal_properties' in key.lower()]

    for key in thermal_properties_list:
        thermal_properties = kwargs[key]
        fe_phonon.append(thermal_properties.get_array('free_energy'))
        entropy.append(thermal_properties.get_array('entropy'))
        cv.append(thermal_properties.get_array('cv'))
        temperatures = thermal_properties.get_array('temperature')



    # Arrange data sorted by volume and transform them to numpy array
    sort_index = np.argsort(volumes)

    volumes = np.array(volumes)[sort_index]
    electronic_energies = np.array(electronic_energies)[sort_index]
    temperatures = np.array(temperatures)
    fe_phonon = np.array(fe_phonon).T[:, sort_index]
    entropy = np.array(entropy).T[:, sort_index]
    cv = np.array(cv).T[:, sort_index]

    opt = np.argmin(electronic_energies)


    # Check minimum energy volume is within the data
    if np.ma.masked_less_equal(volumes, volumes[opt]).mask.all():
        print ('higher volume structures are necessary to compute')
        exit()
    if np.ma.masked_greater_equal(volumes, volumes[opt]).mask.all():
        print ('Lower volume structures are necessary to compute')
        exit()

 #   print volumes.shape
 #   print electronic_energies.shape
 #   print temperatures.shape
 #   print fe_phonon.shape
 #   print cv.shape
 #   print entropy.shape

    qha_output = ArrayData()
    qha_output.set_array('volumes', volumes)
    qha_output.set_array('electronic_energies', electronic_energies)
    qha_output.set_array('temperatures', temperatures)
    qha_output.set_array('fe_phonon', fe_phonon)
    qha_output.set_array('cv', cv)
    qha_output.set_array('entropy', entropy)


    # Calculate QHA
    phonopy_qha = PhonopyQHA(np.array(volumes),
                             np.array(electronic_energies),
                             eos="vinet",
                             temperatures=np.array(temperatures),
                             free_energy=np.array(fe_phonon),
                             cv=np.array(cv),
                             entropy=np.array(entropy),
    #                         t_max=options.t_max,
                             verbose=False)


 #   print phonopy_qha.get_gibbs_temperature()

    qha_output = ArrayData()
    qha_output.set_array('volumes', volumes)
    qha_output.set_array('electronic_energies', electronic_energies)
    qha_output.set_array('temperatures', temperatures)
    qha_output.set_array('fe_phonon', fe_phonon)
    qha_output.set_array('cv', cv)
    qha_output.set_array('entropy', entropy)

    return {'qha_output': qha_output}


    #Get data
    helmholtz_volume = phonopy_qha.get_helmholtz_volume()
    thermal_expansion = phonopy_qha.get_thermal_expansion()
    volume_temperature = phonopy_qha.get_volume_temperature()
    heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
    volume_expansion = phonopy_qha.get_volume_expansion()
    gibbs_temperature = phonopy_qha.get_gibbs_temperature()

    qha_output = ArrayData()
#    qha_output.set_array('temperature', temperatures)
#    qha_output.set_array('helmholtz_volume', np.array(helmholtz_volume))
#    qha_output.set_array('thermal_expansion', np.array(thermal_expansion))
#    qha_output.set_array('volume_temperature', np.array(volume_temperature))
#    qha_output.set_array('heat_capacity_P_numerical', np.array(heat_capacity_P_numerical))
#    qha_output.set_array('volume_expansion', np.array(volume_expansion))
#    qha_output.set_array('gibbs_temperature', np.array(gibbs_temperature))
 #   qha_output.store()

    return {'qha_output': qha_output}



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
wf_complete_list = []
for step_name in ['pressure_expansions', 'collect_data', 'complete', 'pressure_manual_expansions',
                  'pressure_gruneisen']:
    if wf.get_step(step_name):
        wf_complete_list += list(wf.get_step(step_name).get_sub_workflows())

volumes = []
electronic_energies = []
temperatures = []
fe_phonon = []
entropy = []
cv = []

for wf_test in wf_complete_list:
    for pressure in test_pressures:
        if wf_test.get_attribute('pressure') == pressure:
            thermal_properties = wf_test.get_result('thermal_properties')
            optimized_data = wf_test.get_result('optimized_structure_data')
            final_structure = wf_test.get_result('final_structure')

            electronic_energies.append(optimized_data.dict.energy)
            volumes.append(final_structure.get_cell_volume())
            temperatures = thermal_properties.get_array('temperature')
            fe_phonon.append(thermal_properties.get_array('free_energy'))
            entropy.append(thermal_properties.get_array('entropy'))
            cv.append(thermal_properties.get_array('cv'))


from phonopy import PhonopyQHA

sort_index = np.argsort(volumes)

volumes = np.array(volumes)[sort_index]
electronic_energies = np.array(electronic_energies)[sort_index]
temperatures = np.array(temperatures)
fe_phonon = np.array(fe_phonon).T[:, sort_index]
entropy = np.array(entropy).T[:, sort_index]
cv = np.array(cv).T[:, sort_index]

opt = np.argmin(electronic_energies)


# Calculate QHA
phonopy_qha = PhonopyQHA(np.array(volumes),
                         np.array(electronic_energies),
                         eos="vinet",
                         temperatures=np.array(temperatures),
                         free_energy=np.array(fe_phonon),
                         cv=np.array(cv),
                         entropy=np.array(entropy),
                         #                         t_max=options.t_max,
                         verbose=False)

# Get data
qha_temperatures = phonopy_qha._qha._temperatures[:phonopy_qha._qha._max_t_index]
helmholtz_volume = phonopy_qha.get_helmholtz_volume()
thermal_expansion = phonopy_qha.get_thermal_expansion()
volume_temperature = phonopy_qha.get_volume_temperature()
heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
volume_expansion = phonopy_qha.get_volume_expansion()
gibbs_temperature = phonopy_qha.get_gibbs_temperature()


#phonopy_qha.plot_bulk_modulus()
#plt.show()

phonopy_qha.plot_qha()
plt.show()
phonopy_qha.plot_gruneisen_temperature()
plt.show()
phonopy_qha.plot_gibbs_temperature()
plt.show()
phonopy_qha.plot_heat_capacity_P_numerical()
plt.show()

phonopy_qha.write_gibbs_temperature()
phonopy_qha.write_heat_capacity_P_numerical()


qha_output = ArrayData()
qha_output.set_array('temperature', np.array(qha_temperatures))
qha_output.set_array('helmholtz_volume', np.array(helmholtz_volume))
qha_output.set_array('thermal_expansion', np.array(thermal_expansion))
qha_output.set_array('volume_temperature', np.array(volume_temperature))
qha_output.set_array('heat_capacity_P_numerical', np.array(heat_capacity_P_numerical))
qha_output.set_array('volume_expansion', np.array(volume_expansion))
qha_output.set_array('gibbs_temperature', np.array(gibbs_temperature))
#qha_output.store()

# Test to leave something on folder
#        phonopy_qha.plot_pdf_bulk_modulus_temperature()
#        import matplotlib
#       matplotlib.use('Agg')