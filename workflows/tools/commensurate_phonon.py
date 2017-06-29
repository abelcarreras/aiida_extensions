from aiida import load_dbenv

load_dbenv()

from aiida.orm import load_node, load_workflow
from aiida.orm import Code, DataFactory

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
KpointsData = DataFactory('array.kpoints')

import numpy as np


def plot_data(bs):
    import matplotlib.pyplot as plt

    for i, freq in enumerate(bs.get_array('frequencies')):
        plt.plot(bs.get_array('q_path')[i], freq, color='r')

    plt.figure(1)
    plt.axes().get_xaxis().set_ticks([])
    plt.ylabel('Frequency [THz]')
    plt.xlabel('Wave vector')
    plt.xlim([0, bs.get_array('q_path')[-1][-1]])
    plt.axhline(y=0, color='k', ls='dashed')
    plt.suptitle('Phonon band structure')

    if 'labels' in bs.get_arraynames():
        plt.rcParams.update({'mathtext.default': 'regular'})
        labels = bs.get_array('labels')

        labels_e = []
        x_labels = []
        for i, freq in enumerate(bs.get_array('q_path')):
            if labels[i][0] == labels[i - 1][1]:
                labels_e.append('$' + labels[i][0].replace('GAMMA', '\Gamma') + '$')
            else:
                labels_e.append('$' + labels[i - 1][1].replace('GAMMA', '\Gamma') + '/' + labels[i][0].replace('GAMMA',
                                                                                                               '\Gamma') + '$')
            x_labels.append(bs.get_array('q_path')[i][0])
        x_labels.append(bs.get_array('q_path')[-1][-1])
        labels_e.append('$' + labels[-1][1].replace('GAMMA', '\Gamma') + '$')
        labels_e[0] = '$' + labels[0][0].replace('GAMMA', '\Gamma') + '$'

        plt.xticks(x_labels, labels_e, rotation='horizontal')

    # plt.show()

    # Phonon density of states
    dos = wf.get_result('dos')

    frequency = dos.get_array('frequency')
    total_dos = dos.get_array('total_dos')
    partial_dos = dos.get_array('partial_dos')
    partial_symbols = dos.get_array('partial_symbols')

    # Check atom equivalencies
    delete_list = []
    for i, dos_i in enumerate(partial_dos):
        for j, dos_j in enumerate(partial_dos):
            if i < j:
                if np.allclose(dos_i, dos_j) and partial_symbols[i] == partial_symbols[j]:
                    dos_i += dos_j
                    delete_list.append(j)

    partial_dos = np.delete(partial_dos, delete_list, 0)
    partial_symbols = np.delete(partial_symbols, delete_list)

    # print partial_dos
    # print partial_symbols

    plt.figure(2)
    plt.suptitle('Phonon density of states')
    plt.ylabel('Density')
    plt.xlabel('Frequency [THz]')
    plt.ylim([0, np.max(total_dos) * 1.1])

    plt.plot(frequency, total_dos, label='Total DOS')

    for i, dos in enumerate(partial_dos):
        plt.plot(frequency, dos, label='{}'.format(partial_symbols[i]))

    plt.legend()
    # plt.show()

    # Termal properties
    thermal = wf.get_result('thermal_properties')

    free_energy = thermal.get_array('free_energy')
    entropy = thermal.get_array('entropy')
    temperature = thermal.get_array('temperature')
    cv = thermal.get_array('cv')

    plt.figure(3)

    plt.xlabel('Temperature [K]')

    plt.suptitle('Thermal properties')
    plt.plot(temperature, free_energy, label='Free energy (KJ/mol)')
    plt.plot(temperature, entropy, label='entropy (KJ/mol)')
    plt.plot(temperature, cv, label='Cv (J/mol)')

    plt.legend()
    plt.show()


def get_path_using_seekpath(structure, band_resolution=30):
    import seekpath

    cell = structure.cell
    positions = [site.position for site in structure.sites]
    scaled_positions = np.dot(positions, np.linalg.inv(cell))
    numbers = np.unique([site.kind_name for site in structure.sites], return_inverse=True)[1]
    structure2 = (cell, scaled_positions, numbers)
    path_data = seekpath.get_path(structure2)

    labels = path_data['point_coords']

    band_ranges = []
    for set in path_data['path']:
        band_ranges.append([labels[set[0]], labels[set[1]]])

    bands =[]
    for q_start, q_end in band_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)

    return {'ranges': bands,
            'labels': path_data['path']}


def phonopy_calculation_inline(**kwargs):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')
    bands = get_path_using_seekpath(structure)

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'])

    phonon.set_force_constants(force_constants)

    # Normalization factor primitive to unit cell
    normalization_factor = phonon.unitcell.get_number_of_atoms() / phonon.primitive.get_number_of_atoms()

    phonon.set_band_structure(bands['ranges'])

    phonon.set_mesh(phonopy_input['mesh'], is_eigenvectors=True, is_mesh_symmetry=False)
    phonon.set_total_DOS(tetrahedron_method=True)
    phonon.set_partial_DOS(tetrahedron_method=True)

    # get band structure
    band_structure_phonopy = phonon.get_band_structure()
    q_points = np.array(band_structure_phonopy[0])
    q_path = np.array(band_structure_phonopy[1])
    frequencies = np.array(band_structure_phonopy[2])
    band_labels = np.array(bands['labels'])

    # stores band structure
    band_structure = ArrayData()
    band_structure.set_array('q_points', q_points)
    band_structure.set_array('q_path', q_path)
    band_structure.set_array('frequencies', frequencies)
    band_structure.set_array('labels', band_labels)

    # get DOS (normalized to unit cell)
    total_dos = phonon.get_total_DOS() * normalization_factor
    partial_dos = phonon.get_partial_DOS() * normalization_factor

    # Stores DOS data in DB as a workflow result
    dos = ArrayData()
    dos.set_array('frequency', total_dos[0])
    dos.set_array('total_dos', total_dos[1])
    dos.set_array('partial_dos', partial_dos[1])
    dos.set_array('partial_symbols', np.array(phonon.primitive.symbols))

    # THERMAL PROPERTIES (per primtive cell)
    phonon.set_thermal_properties()
    t, free_energy, entropy, cv = phonon.get_thermal_properties()

    # Stores thermal properties (per unit cell) data in DB as a workflow result
    thermal_properties = ArrayData()
    thermal_properties.set_array('temperature', t)
    thermal_properties.set_array('free_energy', free_energy * normalization_factor)
    thermal_properties.set_array('entropy', entropy * normalization_factor)
    thermal_properties.set_array('cv', cv * normalization_factor)

    return {'thermal_properties': thermal_properties, 'dos': dos, 'band_structure': band_structure}




def phonopy_commensurate_inline(**kwargs):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    from phonopy.units import VaspToTHz
    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points, DynmatToForceConstants

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')


    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'])

    phonon.set_force_constants(force_constants)

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()
    dynmat2fc = DynmatToForceConstants(primitive, supercell)
    com_points = dynmat2fc.get_commensurate_points()
    phonon.set_qpoints_phonon(com_points,
                              is_eigenvectors=True)
    frequencies, eigenvectors = phonon.get_qpoints_phonon()

    # Stores DOS data in DB as a workflow result
    commensurate = ArrayData()
    commensurate.set_array('qpoints', com_points)
    commensurate.set_array('frequencies', frequencies)
    commensurate.set_array('eigenvectors', eigenvectors)

    return {'commensurate': commensurate}



def phonopy_commensurate_shifts_inline(**kwargs):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points, DynmatToForceConstants

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')
    r_force_constants = kwargs.pop('r_force_constants').get_array('force_constants')


    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'])

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()


    phonon.set_force_constants(force_constants)

    dynmat2fc = DynmatToForceConstants(primitive, supercell)
    com_points = dynmat2fc.get_commensurate_points()

    phonon.set_qpoints_phonon(com_points,
                              is_eigenvectors=True)
    frequencies_h = phonon.get_qpoints_phonon()[0]


    phonon.set_force_constants(r_force_constants)

    phonon.set_qpoints_phonon(com_points,
                              is_eigenvectors=True)
    frequencies_r = phonon.get_qpoints_phonon()[0]

    shifts = frequencies_r - frequencies_h

    # Stores DOS data in DB as a workflow result
    commensurate = ArrayData()
    commensurate.set_array('qpoints', com_points)
    commensurate.set_array('shifts', shifts)

    return {'commensurate': commensurate}




def phonopy_merge(**kwargs):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    from phonopy.units import VaspToTHz
    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points, DynmatToForceConstants

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()

    harmonic = kwargs.pop('harmonic')
    renormalized = kwargs.pop('renormalized')

    eigenvectors = harmonic.get_array('eigenvectors')
    frequencies = harmonic.get_array('frequencies')
    shifts = renormalized.get_array('shifts')


    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'])

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()



    total_frequencies = frequencies + shifts

    dynmat2fc = DynmatToForceConstants(primitive, supercell)
    dynmat2fc.set_dynamical_matrices(total_frequencies / VaspToTHz, eigenvectors)
    dynmat2fc.run()

    total_force_constants = dynmat2fc.get_force_constants()

    # Stores DOS data in DB as a workflow result
    total_data = ArrayData()
    total_data.set_array('force_constants', total_force_constants)

    return {'final_results': total_data}




# Start script here

# Workflow phonon (at given volume)
wf = load_workflow(431)
parameters = wf.get_parameters()
results = wf.get_results()

inline_params = {'structure': results['final_structure'],
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': results['force_constants']}

harmonic = phonopy_commensurate_inline(**inline_params)



# At reference volume (at T = 0)
wf = load_workflow(432)
parameters = wf.get_parameters()
results_r = wf.get_results()
results_h = wf.get_results()


inline_params = {'structure': results_h['final_structure'],
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': results_h['force_constants'],
                 'r_force_constants': results_r['r_force_constants']}


renormalized = phonopy_commensurate_shifts_inline(**inline_params)





inline_params = {'structure': results_h['final_structure'],
                 'phonopy_input': parameters['phonopy_input'],
                 'harmonic': harmonic,
                 'renormalized': renormalized}

total = phonopy_merge(**inline_params)

print total

inline_params = {'structure': results_h['final_structure'],
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': total['force_constants']}

results = phonopy_calculation_inline(**inline_params)[1]

band = results['band_structure']


# Phonon Band structure plot
plot_data(results['band_structure'])






