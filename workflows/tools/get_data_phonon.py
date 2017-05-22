#!/usr/bin/env python

from aiida import load_dbenv
load_dbenv()

from aiida.orm import load_node, load_workflow

import numpy as np
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy import Phonopy

# Set WorkflowPhonon PK number
#######################
wf = load_workflow(437)
#######################

print ('Results pk: {}'.format(wf.get_result('force_constants').pk))
# 1) Calculated and stored in database

# DOS
if True:
    f = open ('density_of_states', 'w')
    print ('Writing density of states')
    for i,j in zip(wf.get_result('dos').get_array('frequency'), wf.get_result('dos').get_array('total_dos')):
        f.write('{} {}\n'.format(i, j))
    f.close()

# PARTIAL DOS
if True:
    print ('Writing partial density of states')
    f = open('partial_density_of_states', 'w')
    for i,j in zip(wf.get_result('dos').get_array('frequency'),wf.get_result('dos').get_array('partial_dos').T):
        f.write('{} {}\n'.format(i, ' '.join(map(str, j))))
    f.close()

# FORCE CONSTANTS
if True:
    print ('Writing FORCE_CONSTANTS')
    from phonopy.file_IO import write_FORCE_CONSTANTS, write_FORCE_SETS
    force_constants = wf.get_result('force_constants').get_array('force_constants')
    write_FORCE_CONSTANTS(force_constants, filename='FORCE_CONSTANTS')


# 2) Load a complete PHONOPY object
phonopy_input = wf.get_parameters()['phonopy_input'].get_dict()
structure = wf.get_result('final_structure')

bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                            positions=[site.position for site in structure.sites],
                            cell=structure.cell)

phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         primitive_matrix=phonopy_input['primitive'],
                         distance=phonopy_input['distance'])

phonon.set_force_constants(force_constants)

# Save band structure to PDF
bands = []
q_start  = np.array([0.5, 0.5, 0.0])
q_end    = np.array([0.0, 0.0, 0.0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

q_start  = np.array([0.0, 0.0, 0.0])
q_end    = np.array([0.5, 0.0, 0.0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

phonon.set_band_structure(bands)
phonon.write_yaml_band_structure()

plt = phonon.plot_band_structure()
plt.savefig("phonon_band_structure.pdf")
