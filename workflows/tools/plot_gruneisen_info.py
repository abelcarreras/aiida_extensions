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


def get_plot(data, q_path, title=None, ylabel=None, labels=None):
    for i, freq in enumerate(data):
        plt.plot(q_path[i], freq, color='r')

    plt.axes().get_xaxis().set_ticks([])
    plt.ylabel(ylabel)
    plt.xlabel('Wave vector')
    plt.xlim([0, q_path[-1][-1]])
    plt.axhline(y=0, color='k', ls='dashed')
    plt.suptitle(title)

    if labels is not None:
        plt.rcParams.update({'mathtext.default':  'regular' })
        labels_e = []
        x_labels = []
        for i, freq in enumerate(q_path):
            if labels[i][0] == labels[i-1][1]:
                labels_e.append('$'+labels[i][0].replace('GAMMA', '\Gamma')+'$')
            else:
                labels_e.append('$'+labels[i-1][1].replace('GAMMA', '\Gamma')+'/'+labels[i][0].replace('GAMMA', '\Gamma')+'$')
            x_labels.append(q_path[i][0])
        x_labels.append(q_path[-1][-1])
        labels_e.append('$'+labels[-1][1].replace('GAMMA', '\Gamma')+'$')
        labels_e[0]='$'+labels[0][0].replace('GAMMA', '\Gamma')+'$'

        plt.xticks(x_labels, labels_e, rotation='horizontal')


#######################
wf = load_workflow(447)
#######################

# Phonon Band structure

bs = wf.get_result('band_structure')

plt.figure(1)
get_plot(bs.get_array('frequencies'), bs.get_array('q_path'),
         title='Phonon band structure',
         ylabel='Frequency [THz]',
         labels=bs.get_array('labels'))
plt.figure(2)
get_plot(bs.get_array('eigenvalues'), bs.get_array('q_path'),
         title='Eigenvalues',
         labels=bs.get_array('labels'))
plt.figure(3)
get_plot(bs.get_array('gamma'), bs.get_array('q_path'),
         title='Mode Gruneisen parameter',
         ylabel='$\Gamma$',
         labels=bs.get_array('labels'))
plt.show()
exit()


plt.figure(1)
for i, freq in enumerate(bs.get_array('frequencies')):
    plt.plot(bs.get_array('q_path')[i], freq, color='r')

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

plt.figure(2)
for i, freq in enumerate(bs.get_array('eigenvalues')):
    plt.plot(bs.get_array('q_path')[i], freq, color='r')

plt.axes().get_xaxis().set_ticks([])
plt.ylabel('Frequency [THz]')
plt.xlabel('Wave vector')
plt.xlim([0, bs.get_array('q_path')[-1][-1]])
plt.axhline(y=0, color='k', ls='dashed')
plt.suptitle('Eigenvalues')

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


plt.figure(3)
for i, freq in enumerate(bs.get_array('gamma')):
    plt.plot(bs.get_array('q_path')[i], freq, color='r')

plt.axes().get_xaxis().set_ticks([])
plt.ylabel('Frequency [THz]')
plt.xlabel('Wave vector')
plt.xlim([0, bs.get_array('q_path')[-1][-1]])
plt.axhline(y=0, color='k', ls='dashed')
plt.suptitle('Gamma')

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

plt.show()
