import numpy as np

########################################################
#  Common functions for phonon workflows               #
# (should be placed in the same directory of workflow) #
#  Used for workflows: gruneisen, qha                  #
########################################################

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


def thermal_expansion(volumes, electronic_energies, gruneisen, stresses=None, t_max=1000, t_step=10):

    fit_ve = np.polyfit(volumes, electronic_energies, 2)

    test_volumes = np.arange(volumes[0] * 0.8, volumes[0] * 1.2, volumes[0] * 0.01)
    electronic_energies = np.array([np.polyval(fit_ve, i) for i in test_volumes])

    gruneisen.set_thermal_properties(test_volumes, t_min=0, t_max=t_max, t_step=t_step)
    tp = gruneisen.get_thermal_properties()

    normalize = gruneisen.get_phonon().unitcell.get_number_of_atoms() / gruneisen.get_phonon().primitive.get_number_of_atoms()
    free_energy_array = []
    cv_array = []
    entropy_array = []
    total_free_energy_array = []
    for energy, tpi in zip(electronic_energies, tp.get_thermal_properties()):
        temperatures, free_energy, entropy, cv = tpi.get_thermal_properties()
        free_energy_array.append(free_energy)
        entropy_array.append(entropy)
        cv_array.append(cv)
        total_free_energy_array.append(free_energy/normalize + energy)

    total_free_energy_array = np.array(total_free_energy_array)

    fit = np.polyfit(test_volumes, total_free_energy_array, 2)

    min_volume = []
    e_min = []
    for j, t in enumerate(temperatures):
        min_v = -fit.T[j][1] / (2 * fit.T[j][0])
        e_min.append(np.polyval(fit.T[j], min_v))
        min_volume.append(min_v)

    if stresses is not None:

        from scipy.optimize import curve_fit, OptimizeWarning

        try:
            # Fit to an exponential equation
            def fitting_function(x, a, b, c):
                return np.exp(-b * (x + a)) + c

            p_b = 0.1
            p_c = -200
            p_a = -np.log(-p_c) / p_b - volumes[0]

            popt, pcov = curve_fit(fitting_function, volumes, stresses, p0=[p_a, p_b, p_c], maxfev=100000)
            min_stress = fitting_function(min_volume, *popt)

        except OptimizeWarning:
            # Fit to a quadratic equation
            fit_vs = np.polyfit(volumes, stresses, 2)
            min_stress = np.array([np.polyval(fit_vs, v) for v in min_volume])

    else:
        min_stress = None

    return temperatures, min_volume, min_stress


def arrange_band_labels(band_structure):

    substitutions = {'GAMMA': u'\u0393'
                     }

    def replace_list(text_string, substitutions):

        for item in substitutions.iteritems():
            text_string = text_string.replace(item[0], item[1])

        return text_string

    labels_array = band_structure.get_array('labels')

    labels = []
    labels_positions = []
    for i, freq in enumerate(band_structure.get_array('q_path')):
        if labels_array[i][0] == labels_array[i-1][1]:
            labels.append(replace_list(labels_array[i][0],substitutions))
        else:
            labels.append(replace_list(labels_array[i-1][1]+'/'+labels_array[i][0], substitutions))
        labels_positions.append(band_structure.get_array('q_path')[i][0])
    labels_positions.append(band_structure.get_array('q_path')[-1][-1])
    labels.append(replace_list(labels_array[-1][1], substitutions))
    labels[0] = replace_list(labels_array[0][0], substitutions)

    return labels_positions, labels


def write_unicode_file(labels_positions, labels):
    import StringIO
    output = StringIO.StringIO()

    for i, j in zip(labels_positions, labels):
       output.write(u'{0:12.8f}       {1}\n'.format(i, j).encode('utf-8'))
    output.seek(0)

    return output

def get_file_from_txt(text):

    import StringIO
    output = StringIO.StringIO()
    output.write(text)
    output.seek(0)

    return output


def smearing_function_mesh(X, Y, frequencies, gruneisen, sigma=0.1):

    frequencies = frequencies.reshape(-1)
    gruneisen = gruneisen.reshape(-1)

    def gaussian(X, Y, sigma, freq, grune):
        result = 1.0/np.sqrt(2*np.pi*sigma**2) * np.exp(-((X-freq)**2 + (Y-grune)**2)/(2*sigma**2))
        return result


    total = np.zeros_like(X)
    for freq, grune in zip(frequencies, gruneisen):
        total += gaussian(X,Y, sigma, freq, grune)

    return total/len(frequencies)


# convert numpy string into web page ready text file
def get_file_from_numpy_array(data, text_list=None):
    import StringIO
    output = StringIO.StringIO()
    if text_list is None:
        output.write('# No caption\n')
    else:
        output.write('       '.join(text_list) + '\n')

    for line in np.array(data).astype(str):
        output.write('       '.join(line) + '\n')
    output.seek(0)
    return output


def get_data_info(structure):

    pmg_structure = structure.get_pymatgen_structure()
    formula = pmg_structure.formula
    space_group = pmg_structure.get_space_group_info()
    lattice_vectors = pmg_structure.lattice.matrix
    positions = pmg_structure.frac_coords
    species = pmg_structure.species
    volume = pmg_structure.volume

    info_data = ''
    info_data += '<b>Formula:</b> {}\n'.format(formula)
    info_data += '<br><b>Space group:</b> {}   #{}\n'.format(*space_group)

    info_data += '\n'
    info_data += '<br><br><b>Lattice vectors (Angstroms)</b>\n'
    info_data += ('<br>{0:10.8f}  {1:10.8f}  {2:10.8f}\n'.format(*lattice_vectors[0]) +
                  '<br>{0:10.8f}  {1:10.8f}  {2:10.8f}\n'.format(*lattice_vectors[1]) +
                  '<br>{0:10.8f}  {1:10.8f}  {2:10.8f}\n'.format(*lattice_vectors[2]))
    info_data += '\n'
    info_data += '<br><br><b>Positions (frac. coord)</b>\n'
    for i, xyz in enumerate(positions):
        info_data += ('<br>{}  '.format(species[i]) + '{0:10.8f}  {1:10.8f}  {2:10.8f}\n'.format(*xyz))
    info_data += '\n'
    info_data += '<br><br><b>Volume:</b> {} Angstroms<sup>3</sup>\n'.format(volume)

    return info_data


def get_helmholtz_volume_from_phonopy_qha(phonopy_qha, thin_number=10):

    from numpy import max, min
    self = phonopy_qha._qha
    volume_points = np.linspace(min(self._volumes),
                                max(self._volumes),
                                201)
    min_volumes = []
    min_energies = []

    volumes = self._volumes
    selected_energies = []
    energies_points = []

    for i, t in enumerate(self._temperatures[:self._max_t_index]):
        if i % thin_number == 0:
            min_volumes.append(self._equiv_volumes[i])
            min_energies.append(self._equiv_energies[i])

            selected_energies.append(self._free_energies[i])

            energies_points.append(self._eos(volume_points, *self._equiv_parameters[i]))

    return {'fit': (volume_points, np.array(energies_points)),
            'points':  (volumes, np.array(selected_energies)),
            'minimum': (min_volumes, min_energies)}


# Write to files
def get_FORCE_CONSTANTS_txt(force_constants_object):

    force_constants = force_constants_object.get_array('force_constants')

    # Write FORCE CONSTANTS
    force_constants_txt = '{0}\n'.format(len(force_constants))
    for i, fc in enumerate(force_constants):
        for j, atomic_fc in enumerate(fc):
            force_constants_txt += '{0} {1}\n'.format(i, j)
            for line in atomic_fc:
                force_constants_txt += '{0:20.16f} {1:20.16f} {2:20.16f}\n'.format(*line)

    return force_constants_txt


def structure_to_poscar(structure):
    poscar = ' '.join(np.unique([site.kind_name for site in structure.sites]))
    poscar += '\n1.0\n'
    cell = structure.cell
    for row in cell:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*row)
    poscar += ' '.join(np.unique([site.kind_name for site in structure.sites])) + '\n'
    poscar += str(len(structure.sites)) + '\n'
    poscar += 'Cartesian\n'
    for site in structure.sites:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*site.position)

    return poscar


if __name__ == '__main__':

    def gaussian(X, Y, sigma, freq, grune):
        result = 1.0/np.sqrt(2*np.pi*sigma**2) * np.exp(-((X-freq)**2 + (Y-grune)**2)/(2*sigma**2))
        return result

    x = np.arange(-2, 2, 0.1)
    y = np.arange(-2, 2, 0.1)
    X, Y = np.meshgrid(x,y)

    frequencies = np.sin(np.linspace(-6.3, 6.3, 1000))
    gruneisen = np.cos(np.linspace(-6.3, 6.3, 1000))

    Z = smearing_function_mesh(X, Y, frequencies, gruneisen)

    #Z = gaussian(X, Y, 0.1, 0, 0)
    import matplotlib.pyplot as plt

    plt.contour(X, Y, Z)
    plt.show()
    exit()

    plt.plot(np.arange(-10, 10, 0.1), [gaussian(x, 0, 0.5, [0, 0]) for x in np.arange(-10, 10, 0.1)])
    plt.show()
    exit()


