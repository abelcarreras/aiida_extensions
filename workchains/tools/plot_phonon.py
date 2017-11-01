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
import sys

if len(sys.argv) < 2:
    print ('use: plot_phonon pk_number')
    exit()

# Set WorkChain PhononPhonopy PK number
################################
wc = load_node(int(sys.argv[1]))
################################

# Phonon Band structure
bs = wc.out.band_structure

plt.figure(1)
for dist, freq in zip(bs.get_distances(), bs.get_frequencies()):
    plt.plot(dist, freq, color='r')

plt.axes().get_xaxis().set_ticks([])
plt.ylabel('Frequency [THz]')
plt.xlabel('Wave vector')
plt.xlim([0, bs.get_distances()[-1][-1]])
plt.axhline(y=0, color='k', ls='dashed')
plt.suptitle('Phonon band structure')

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular'})
    labels, label_positions = bs.get_formatted_labels_matplotlib()
    plt.xticks(label_positions, labels, rotation='horizontal')

#plt.show()

# Phonon density of states
dos = wc.out.dos

total_dos = dos.get_dos()
frequency = dos.get_frequencies()
partial_dos = dos.get_partial_dos()
partial_symbols = dos.get_atom_labels()

plt.figure(2)
plt.suptitle('Phonon density of states')
plt.ylabel('Density')
plt.xlabel('Frequency [THz]')
plt.ylim([0, np.max(total_dos) * 1.1])

plt.plot(frequency, total_dos, label='Total DOS')

for dos, symbol in zip(partial_dos, partial_symbols):
    plt.plot(frequency, dos, label='{}'.format(symbol))

plt.legend()
#plt.show()

# Thermal properties
thermal = wc.out.thermal_properties

free_energy = thermal.get_array('free_energy')
entropy = thermal.get_array('entropy')
temperature = thermal.get_array('temperature')
cv = thermal.get_array('cv')

plt.figure(3)

plt.xlabel('Temperature [K]')

plt.suptitle('Thermal properties (per unit cell)')
plt.plot(temperature, free_energy, label='Free energy (KJ/mol)')
plt.plot(temperature, entropy, label='entropy (KJ/mol)')
plt.plot(temperature, cv, label='Cv (J/mol)')

plt.legend()
plt.show()


bs = wc.out.band_structure


labels, indices = bs.get_formatted_labels_blocks()

from matplotlib import gridspec

width = []
for ind in indices:
    width.append(bs.get_distances(band=ind[-1])[-1] - bs.get_distances(band=ind[0])[0])

print ('width', width)

gs = gridspec.GridSpec(1, 3, width_ratios=width, wspace=0.05)
print gs
plt.figure(4)
ylim = None
for j, index in enumerate(indices):
    print ('j', j)
    ax1 = plt.subplot(gs[j])

    #ax1 = plt.subplot(1, len(labels), j+1)
    for i in index:
        freq = bs.get_frequencies(band=i)
        dist = bs.get_distances(band=i)
        ax1.plot(dist, freq, color='r')
    print [bs.get_bands(band=index[0])[0], bs.get_bands(band=index[-1])[-1]]
    if j !=0:
        ax1.axes.get_yaxis().set_visible(False)


    #plt.ylim([0, 25])
    plt.axhline(y=0.0, color='b', linestyle='--')
    plt.xlim([bs.get_distances(band=index[0])[0], bs.get_distances(band=index[-1])[-1]])
    position = [bs.get_distances(band=i)[0] for i in index] + [bs.get_distances(band=index[-1])[-1]]
    plt.rcParams.update({'mathtext.default': 'regular'})
    plt.xticks(position, labels[j], rotation='horizontal')

plt.figure(figsize=(20, 10))
plt.autoscale(enable=True, axis='y')
plt.show()

exit()

ax_list = plt.subplots(*range(len(labels)))[1]
ax1, ax2 = plt.subplots(1, 2)[1]

# Two subplots, unpack the axes array immediately
plt.rcParams.update({'mathtext.default': 'regular'})
labels, label_positions = bs.get_formatted_labels_matplotlib()
a = bs.get_labels()

plt.xticks(label_positions, labels, rotation='horizontal')
ax1.plot(bs.get_distances()[0], bs.get_frequencies()[0])

plt.xticks(label_positions, labels, rotation='horizontal')
ax2.plot(bs.get_distances()[1], bs.get_frequencies()[1])

plt.show()
