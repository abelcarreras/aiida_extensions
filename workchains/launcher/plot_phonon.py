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

# Set WorkflowPhonon PK number
########################
wc = load_node(488914)
########################

# Phonon Band structure
bs = wc.out.band_structure
for dist, freq in zip(bs.get_distances(), bs.get_frequencies()):
    plt.plot(dist, freq, color='r')

plt.figure(1)
plt.axes().get_xaxis().set_ticks([])
plt.ylabel('Frequency [THz]')
plt.xlabel('Wave vector')
plt.xlim([0, bs.get_distances()[-1][-1]])
plt.axhline(y=0, color='k', ls='dashed')
plt.suptitle('Phonon band structure')


print bs.get_labels()

if bs.get_labels() is not None:
    plt.rcParams.update({'mathtext.default': 'regular' })
    labels = bs.get_labels()

    labels_e = []
    x_labels = []
    for i, freq in enumerate(bs.get_distances()):
        if bs.get_labels(i)[0] == bs.get_labels(i-1)[1]:
            labels_e.append('$'+ bs.get_labels(i)[0].replace('GAMMA', '\Gamma')+'$')
        else:
            labels_e.append('$'+ bs.get_labels(i-1)[1].replace('GAMMA', '\Gamma')+'/'+ bs.get_labels(i)[0].replace('GAMMA', '\Gamma')+'$')
        x_labels.append(bs.get_distances(band=i)[0])
    x_labels.append(bs.get_distances(band=-1)[-1])
    labels_e.append('$'+ bs.get_labels(-1)[1].replace('GAMMA', '\Gamma')+'$')
    labels_e[0]='$'+ bs.get_labels(0)[0].replace('GAMMA', '\Gamma')+'$'

    plt.xticks(x_labels, labels_e, rotation='horizontal')

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
