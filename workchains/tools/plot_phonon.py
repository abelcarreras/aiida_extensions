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


bs = wc.out.band_structure

labels, indices, widths, ranges, positions = bs.get_plot_helpers()
gs = gridspec.GridSpec(1, len(widths), width_ratios=widths, wspace=0.05)

plt.figure(4)

plt.rcParams.update({'mathtext.default': 'regular'})

for j, index in enumerate(indices):
    ax1 = plt.subplot(gs[j])
    for i in index:
        ax1.plot(bs.get_distances(band=i),
                 bs.get_frequencies(band=i),
                 color='r')
    if j != 0:
        ax1.axes.get_yaxis().set_visible(False)

    plt.axhline(y=0.0, color='b', linestyle='--')
    plt.ylabel('Frequency (THz)')
    plt.xlim(ranges[j])
    plt.xticks(positions[j], labels[j], rotation='horizontal')

plt.suptitle('Phonon band structure')
plt.autoscale(enable=True, axis='y')
plt.figtext(0.5, 0.02, 'Wave vector', ha='center')
plt.show()

