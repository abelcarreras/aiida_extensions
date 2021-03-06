from aiida import load_dbenv
load_dbenv()

from aiida.orm import load_node, load_workflow
from aiida.orm import Code, DataFactory

import matplotlib.pyplot as plt
import numpy as np

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
KpointsData = DataFactory('array.kpoints')

def get_plot(band_data, q_path, title='', ylabel='', labels=None, q_points=None, freq_tolerance=1e-5):

    for i, (data, p) in enumerate(zip(band_data, q_path)):
        if q_points is not None:
            q_norm = np.linalg.norm(q_points[i], axis=1)
            indices = np.where(q_norm > freq_tolerance)[0]
            plt.plot(p[indices], data[indices], color='r')
        else:
            plt.plot(p, data, color='r')


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
        for i, data in enumerate(q_path):
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
wf = load_workflow(276)
#######################

# Band structure

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
get_plot(bs.get_array('gruneisen'), bs.get_array('q_path'),
         title='Mode Gruneisen parameter',
         ylabel='$\gamma$',
         labels=bs.get_array('labels'),
         q_points=bs.get_array('q_points'))

# Mesh
plt.figure(4)

mesh = wf.get_result('mesh')

plt.xlabel('Frequency [THz]')
plt.ylabel('$\gamma$')
plt.suptitle('Mode Gruneisen parameter')

for (g, freq) in zip(mesh.get_array('frequencies').T, mesh.get_array('gruneisen').T):
    plt.plot(g, freq, marker='o', linestyle='None', markeredgecolor='black', color='red')

# Prediction

prediction = wf.get_result('thermal_expansion_prediction')

temperatures = prediction.get_array('temperatures')
stresses = prediction.get_array('stresses')
volumes = prediction.get_array('volumes')

plt.figure(5)
plt.xlabel('Temperature [K]')
plt.ylabel('Stress [Kbar]')
plt.suptitle('Prediction stress')
plt.plot(temperatures, stresses)

plt.figure(6)
plt.xlabel('Temperature [K]')
plt.ylabel('Volume [A^3]')
plt.suptitle('Prediction Volume')
plt.plot(temperatures, volumes)


plt.figure(7)
plt.xlabel('Volume [A^3]')
plt.ylabel('Stress [Kbar]')
plt.suptitle('Volume - stress')
plt.plot(volumes, stresses)


plt.show()



