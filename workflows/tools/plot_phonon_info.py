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
#######################
wf = load_workflow(437)
#######################

# Phonon Band structure
bs = wf.get_result('band_structure')
for i, freq in enumerate(bs.get_array('frequencies')):
    plt.plot(bs.get_array('q_path')[i], freq, color='r')

plt.figure(1)
plt.axes().get_xaxis().set_ticks([])
plt.ylabel('Frequency [THz]')
plt.xlabel('Wave vector')
plt.xlim([0, bs.get_array('q_path')[-1][-1]])
plt.axhline(y=0, color='k', ls='dashed')
plt.suptitle('Phonon band structure')

if 'labels' in bs.get_arraynames():
    plt.rcParams.update({'mathtext.default':  'regular' })
    labels = bs.get_array('labels')

    labels_e = []
    x_labels = []
    for i, freq in enumerate(bs.get_array('q_path')):
        if labels[i][0] == labels[i-1][1]:
            labels_e.append('$'+labels[i][0].replace('GAMMA', '\Gamma')+'$')
        else:
            labels_e.append('$'+labels[i-1][1].replace('GAMMA', '\Gamma')+'/'+labels[i][0].replace('GAMMA', '\Gamma')+'$')
        x_labels.append(bs.get_array('q_path')[i][0])
    x_labels.append(bs.get_array('q_path')[-1][-1])
    labels_e.append('$'+labels[-1][1].replace('GAMMA', '\Gamma')+'$')
    labels_e[0]='$'+labels[0][0].replace('GAMMA', '\Gamma')+'$'

    plt.xticks(x_labels, labels_e, rotation='horizontal')

#plt.show()

# Phonon density of states
dos = wf.get_result('dos')

frequency = dos.get_array('frequency')
total_dos = dos.get_array('total_dos')
partial_dos = dos.get_array('partial_dos')
partial_symbols = dos.get_array('partial_symbols')


# Check atom equivalences
delete_list = []
for i, dos_i in enumerate(partial_dos):
    for j, dos_j  in enumerate(partial_dos):
        if i < j:
            if np.allclose(dos_i, dos_j) and partial_symbols[i] == partial_symbols[j]:
                dos_i += dos_j
                delete_list.append(j)

partial_dos = np.delete(partial_dos, delete_list, 0)
partial_symbols = np.delete(partial_symbols, delete_list)


plt.figure(2)
plt.suptitle('Phonon density of states')
plt.ylabel('Density')
plt.xlabel('Frequency [THz]')
plt.ylim([0, np.max(total_dos)*1.1])

plt.plot(frequency, total_dos, label='Total DOS')

for i, dos in enumerate(partial_dos):
    plt.plot(frequency, dos, label='{}'.format(partial_symbols[i]))

plt.legend()
#plt.show()

# Termal properties
thermal = wf.get_result('thermal_properties')

free_energy = thermal.get_array('free_energy')
entropy = thermal.get_array('entropy')
temperature = thermal.get_array('temperature')
cv = thermal.get_array('cv')

plt.figure(3)

plt.xlabel('Temperature [K]')

plt.suptitle('Thermal properties')
plt.plot(temperature, free_energy, label='Free energy (KJ/mol)')
plt.plot(temperature, entropy, label='entropy (KJ/mol)')
plt.plot(temperature, cv, label='Cv (J/mol)')

plt.legend()
plt.show()
