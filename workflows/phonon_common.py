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

    free_energy_array = []
    cv_array = []
    entropy_array = []
    total_free_energy_array = []
    for energy, tpi in zip(electronic_energies, tp.get_thermal_properties()):
        temperatures, free_energy, entropy, cv = tpi.get_thermal_properties()
        free_energy_array.append(free_energy)
        entropy_array.append(entropy)
        cv_array.append(cv)
        total_free_energy_array.append(free_energy + energy)

    total_free_energy_array = np.array(total_free_energy_array)

    fit = np.polyfit(test_volumes, total_free_energy_array, 2)

    min_volume = []
    e_min = []
    for j, t in enumerate(temperatures):
        min_v = -fit.T[j][1] / (2 * fit.T[j][0])
        e_min.append(np.polyval(fit.T[j], min_v))
        min_volume.append(min_v)

    if stresses is not None:

        from scipy.optimize import curve_fit

        # Fit to an exponential equation
        def fitting_function(x, a, b, c):
            return np.exp(-b * (x + a)) + c

        p_b = 0.1
        p_c = -200
        p_a = -np.log(-p_c) / p_b - volumes[0]

        popt, pcov = curve_fit(fitting_function, volumes, stresses, p0=[p_a, p_b, p_c], maxfev=100000)
        min_stress = fitting_function(min_volume, *popt)

        # Fit to a quadratic equation
        # fit_vs = np.polyfit(volumes, stresses, 2)
        # min_stress = np.array([np.polyval(fit_vs, v) for v in min_volume])
    else:
        min_stress = None

    return temperatures, min_volume, min_stress


