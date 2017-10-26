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

# Set WorkChain GruneisenPhonopy PK number
########################
wc = load_node(5852)
########################


# Phonon Band structure
bs = wc.out.band_structure
labels, label_positions = bs.get_formatted_labels_matplotlib()
plt.rcParams.update({'mathtext.default': 'regular'})
plt.xticks(label_positions, labels, rotation='horizontal')

plt.figure(1)
for dist, freq in zip(bs.get_distances(), bs.get_frequencies()):
    plt.plot(dist, freq, color='r',
             title='Phonon band structure',
             ylabel='Frequency [THz]')

plt.figure(2)
for dist, freq in zip(bs.get_distances(), bs.get_eigenvalues()):
    plt.plot(dist, freq, color='r',
             title='Eigenvalues',
             ylabel='Eigenvalues')


plt.figure(3)
for dist, freq in zip(bs.get_distances(), bs.get_gamma()):
    plt.plot(dist, freq, color='r',
             title='Mode Gruneisen parameter',
             ylabel='$\gamma$')

# Mesh
plt.figure(4)

mesh = wc.out.mesh

plt.xlabel('Frequency [THz]')
plt.ylabel('$\gamma$')
plt.suptitle('Mode Gruneisen parameter')

for (g, freq) in zip(mesh.get_array('frequencies').T, mesh.get_array('gruneisen').T):
    plt.plot(g, freq, marker='o', linestyle='None', markeredgecolor='black', color='red')

plt.show()
