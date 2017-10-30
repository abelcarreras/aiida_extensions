from aiida import load_dbenv
load_dbenv()

from aiida.orm import load_node, load_workflow
from aiida.orm import Code, DataFactory

import matplotlib.pyplot as plt

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
KpointsData = DataFactory('array.kpoints')

import sys
import numpy as np

if len(sys.argv) < 2:
    print ('use: plot_gruneisen pk_number')
    exit()

# Set WorkChain GruneisenPhonopy PK number
################################
wc = load_node(int(sys.argv[1]))
################################

gamma_cutoff = 0.1

# Phonon Band structure
bs = wc.out.band_structure

plt.figure(1)
for dist, freq in zip(bs.get_distances(), bs.get_frequencies()):
    plt.plot(dist, freq, color='r')
plt.ylabel('Frequency [THz]')
plt.title('Phonon band structure')

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular'})
    labels, label_positions = bs.get_formatted_labels_matplotlib()
    plt.xticks(label_positions, labels, rotation='horizontal')


plt.figure(2)
for dist, freq in zip(bs.get_distances(), bs.get_eigenvalues()):
    plt.plot(dist, freq, color='r')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues')

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular'})
    labels, label_positions = bs.get_formatted_labels_matplotlib()
    plt.xticks(label_positions, labels, rotation='horizontal')

plt.figure(3)
bands = bs.get_bands()
#for dist, freq in zip(bs.get_distances(), bs.get_gamma()):
for i, dist in enumerate(bs.get_distances()):
    gamma = bs.get_gamma(band=i)
    q_points = bs.get_bands(band=i)
    mask = np.where(np.linalg.norm(q_points, axis=1) > gamma_cutoff)

    plt.plot(dist[mask], gamma[mask], color='r')
plt.ylabel('$\gamma$')
plt.title('Mode Gruneisen parameter')

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular'})
    labels, label_positions = bs.get_formatted_labels_matplotlib()
    plt.xticks(label_positions, labels, rotation='horizontal')


# Mesh
mesh = wc.out.mesh

plt.figure(4)
#for g, freq in zip(mesh.get_array('frequencies').T, mesh.get_array('gruneisen').T):

for i, freq in enumerate(mesh.get_array('frequencies').T):
    gamma = mesh.get_array('gamma').T[i]
    #q_points = mesh.get_array('q_points').T
    #print q_points

    #mask = np.where(np.linalg.norm(q_points, axis=1) > gamma_cutoff)
    plt.plot(gamma, freq, marker='o', linestyle='None', markeredgecolor='black', color='red')
plt.xlabel('Frequency [THz]')
plt.ylabel('$\gamma$')
plt.title('Mode Gruneisen parameter (mesh)')

plt.show()
