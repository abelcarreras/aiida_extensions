from aiida import load_dbenv
load_dbenv()

from aiida.orm import load_node, load_workflow
from aiida.orm import Code, DataFactory

import matplotlib.pyplot as plt
from matplotlib import gridspec

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

labels, indices, widths, ranges, positions = bs.get_plot_helpers()
gs = gridspec.GridSpec(1, len(widths), width_ratios=widths, wspace=0.05)

plt.figure(1)

plt.rcParams.update({'mathtext.default': 'regular'})

for j, index in enumerate(indices):
    ax1 = plt.subplot(gs[j])

    plt.gca().set_color_cycle(None)
    for i in index:
        ax1.plot(bs.get_distances(band=i),
                 bs.get_frequencies(band=i),
                 #color='r'
                 )
    if j != 0:
        ax1.axes.get_yaxis().set_visible(False)

    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.1)
    plt.ylabel('Frequency (THz)')
    plt.xlim(ranges[j])
    plt.xticks(positions[j], labels[j], rotation='horizontal')

plt.suptitle('Phonon band structure')
plt.autoscale(enable=True, axis='y')
plt.figtext(0.5, 0.02, 'Wave vector', ha='center')


plt.figure(2)

plt.rcParams.update({'mathtext.default': 'regular'})

for j, index in enumerate(indices):
    ax1 = plt.subplot(gs[j])

    plt.gca().set_color_cycle(None)
    for i in index:
        q_points = bs.get_bands(band=i)
        mask = np.where(np.linalg.norm(q_points, axis=1) > gamma_cutoff)

        ax1.plot(bs.get_distances(band=i)[mask],
                 bs.get_gamma(band=i)[mask],
                 # color='r'
                 )

    if j != 0:
        ax1.axes.get_yaxis().set_visible(False)
        ax0 = plt.subplot(gs[j-1])
        #ax1.set_ylim(ax0.get_ylim())

    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.1)
    plt.ylabel('$\gamma$')
    plt.xlim(ranges[j])
    #plt.ylim([y_max - y_len*0.1, y_max + y_len*0.1])
    plt.xticks(positions[j], labels[j], rotation='horizontal')

plt.suptitle('Mode Gruneisen parameter')
plt.autoscale(enable=True, axis='y')
plt.figtext(0.5, 0.02, 'Wave vector', ha='center')



plt.show()




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
    plt.xlim([0, bs.get_distances()[-1][-1]])

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
    plt.xlim([0, bs.get_distances()[-1][-1]])

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
