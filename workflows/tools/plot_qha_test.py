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

#######################
wf = load_workflow(512)
#######################

wf_list = list(wf.get_step('pressure_expansions').get_sub_workflows())[:-1]
wf_list += list(wf.get_step('start').get_sub_workflows())


print wf_list[0].get_result('band_structure').get_array('q_points')[1][-1]

band = 4
volume = []
energy = []
grune_qpoint = []
for wf_grune in wf_list:
     volume.append(wf_grune.get_result('final_structure').get_cell_volume())
     energy.append(wf_grune.get_result('optimized_structure_data').dict.energy)
     grune_qpoint.append(wf_grune.get_result('band_structure').get_array('gruneisen')[band])

q_path  = wf_grune.get_result('band_structure').get_array('q_path')[band]


sort_index = np.argsort(volume)

volume = np.array(volume)[sort_index]
energy = np.array(energy)[sort_index]
grune_qpoint = np.array(grune_qpoint)[sort_index]

for v, e in zip(volume, energy):
    print ('{}  {}'.format(v, e))



grune_qpoint = np.sort(grune_qpoint, axis=2)

#plt.plot(volume, grune_qpoint)
#plt.show()


#band_index = 0
X, Y = np.meshgrid(volume, q_path)
#Z = np.array(np.array(grune_qpoint).T[0])

for i,Z in  enumerate(np.array(np.array(grune_qpoint).T)):
   plt.figure(i+1)
   plt.xlabel('Volume[A^3]')
   plt.ylabel('q path')
   CS = plt.contour(X, Y, Z)
   plt.clabel(CS, inline=1, fontsize=10)
   plt.title('Mode {}'.format(i+1))

plt.show()


from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Volume [A^3]')
ax.set_ylabel('q path')
ax.set_zlabel('Mode Gruneisen parameter')

colors = ['r', 'g', 'b', 'y', 'r', 'g']
for i,Z in enumerate(np.array(np.array(grune_qpoint).T)):
    ax.plot_wireframe(X, Y, Z, rstride=2, cstride=1, color=np.roll(colors,i)[0])
