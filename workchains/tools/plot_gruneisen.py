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

gamma_cutoff = 0.01

# Phonon Band structure
bs = wc.out.band_structure

plt.figure(1)
for dist, freq in zip(bs.get_distances(), bs.get_frequencies()):
    plt.plot(dist,
             freq,
             #color='r'
             )
    plt.gca().set_color_cycle(None)

plt.ylabel('Frequency [THz]')
plt.title('Phonon band structure')

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular'})
    labels, label_positions = bs.get_formatted_labels_matplotlib()
    plt.xticks(label_positions, labels, rotation='horizontal')

plt.figure(2)
for i, dist in enumerate(bs.get_distances()):
    gamma = bs.get_gamma(band=i)
    q_points = bs.get_bands(band=i)
    mask = np.where(np.linalg.norm(q_points, axis=1) > gamma_cutoff)

    plt.plot(dist[mask],
             gamma[mask],
             # color='r'
             )
    plt.gca().set_color_cycle(None)

plt.ylabel('$\gamma$')
plt.title('Mode Gruneisen parameter')

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular'})
    labels, label_positions = bs.get_formatted_labels_matplotlib()
    plt.xticks(label_positions, labels, rotation='horizontal')


# Mesh
mesh = wc.out.mesh

plt.figure(3)
q_points = mesh.get_array('q_points')
mask = np.where(np.linalg.norm(q_points, axis=1) > gamma_cutoff)

for gamma, freq in zip( mesh.get_array('gruneisen').T,
                        mesh.get_array('frequencies').T):
    plt.plot(freq[mask], gamma[mask],
             marker='o',
             linestyle='None',
             markeredgecolor='black',
             # color='red'
             )
plt.xlabel('Frequency [THz]')
plt.ylabel('$\gamma$')
plt.title('Mode Gruneisen parameter (mesh)')

plt.show()
