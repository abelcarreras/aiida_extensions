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
wc = load_node(6183)
########################

# Phonon Band structure
bs = wc.out.band_structure

print bs.get_frequencies().shape
for dist, freq in zip(bs.get_distances(), bs.get_frequencies()):
    plt.plot(dist, freq, color='r')

plt.figure(1)
plt.axes().get_xaxis().set_ticks([])
plt.ylabel('Frequency [THz]')
plt.xlabel('Wave vector')
plt.xlim([0, bs.get_distances()[-1][-1]])
plt.axhline(y=0, color='k', ls='dashed')
plt.suptitle('Phonon band structure')

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular'})
    labels, label_positions = bs.get_formatted_labels_matplotlib()
    #plt.xticks(label_positions, labels, rotation='horizontal')

plt.show()


# Phonon Band structure
bs = wc.out.band_structure

print bs.get_frequencies().shape

labels, label_positions = bs.get_formatted_labels_matplotlib()
plt.rcParams.update({'mathtext.default': 'regular'})
plt.xticks(label_positions, labels, rotation='horizontal')
plt.xlabel('Wave vector')

plt.figure(1)
for dist, freq in zip(bs.get_distances(), bs.get_frequencies()):
    plt.plot(dist, freq, color='r')
plt.ylabel('Frequency [THz]')
plt.title('Phonon band structure')
plt.show()
exit()

plt.figure(2)
for dist, freq in zip(bs.get_distances(), bs.get_eigenvalues()):
    plt.plot(dist, freq, color='r')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues')

plt.figure(3)
for dist, freq in zip(bs.get_distances(), bs.get_gamma()):
    plt.plot(dist, freq, color='r')
plt.ylabel('$\gamma$')
plt.title('Mode Gruneisen parameter')

# Mesh

mesh = wc.out.mesh
plt.xlabel('Frequency [THz]')
plt.ylabel('$\gamma$')
plt.suptitle('Mode Gruneisen parameter')

plt.figure(4)
for (g, freq) in zip(mesh.get_array('frequencies').T, mesh.get_array('gruneisen').T):
    plt.plot(g, freq, marker='o', linestyle='None', markeredgecolor='black', color='red')

plt.show()
