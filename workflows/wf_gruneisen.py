from aiida.orm import Code, DataFactory
from aiida.orm.workflow import Workflow
from aiida.orm.calculation.inline import make_inline

from aiida.workflows.wf_phonon import WorkflowPhonon

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')

import numpy as np

def get_path_using_seekpath(structure, band_resolution=30):
    import seekpath

    cell = structure.cell
    positions = [site.position for site in structure.sites]
    scaled_positions = np.dot(positions, np.linalg.inv(cell))
    numbers = np.unique([site.kind_name for site in structure.sites], return_inverse=True)[1]
    structure2 = (cell, scaled_positions, numbers)
    path_data = seekpath.get_path(structure2)

    labels = path_data['point_coords']

    band_ranges = []
    for set in path_data['path']:
        band_ranges.append([labels[set[0]], labels[set[1]]])

    bands =[]
    for q_start, q_end in band_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)

    return {'ranges': bands,
            'labels': path_data['path']}


def get_phonon(structure, force_constants, phonopy_input):
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    # Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols=[site.kind_name for site in structure.sites],
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)

    phonon = Phonopy(bulk,
                     phonopy_input['supercell'],
                     primitive_matrix=phonopy_input['primitive'],
                     distance=phonopy_input['distance'])

    phonon.set_force_constants(force_constants)

    return phonon


@make_inline
def phonopy_gruneisen_inline(**kwargs):
    from phonopy import PhonopyGruneisen

    phonopy_input = kwargs.pop('phonopy_input').get_dict()

    structure_origin = kwargs.pop('structure_origin')

    phonon_plus = get_phonon(kwargs.pop('structure_plus'),
                             kwargs.pop('force_constants_plus').get_array('force_constants'),
                             phonopy_input['supercell'])

    phonon_minus = get_phonon(kwargs.pop('structure_minus'),
                              kwargs.pop('force_constants_minus').get_array('force_constants'),
                              phonopy_input['supercell'])

    phonon_origin = get_phonon(structure_origin,
                               kwargs.pop('force_constants_origin').get_array('force_constants'),
                               phonopy_input['supercell'])

    gruneisen = PhonopyGruneisen(phonon_origin,  # equilibrium
                                 phonon_plus,  # plus
                                 phonon_minus)  # minus

    bands = get_path_using_seekpath(structure_origin)

    gruneisen.set_band_structure(bands['ranges'], 51)

    band_structure_phonopy = gruneisen.get_band_structure()
    q_points = np.array(band_structure_phonopy[0])
    q_path = np.array(band_structure_phonopy[1])
    frequencies = np.array(band_structure_phonopy[2])
    band_labels = np.array(bands['labels'])

    # stores band structure
    band_structure = ArrayData()
    band_structure.set_array('q_points', q_points)
    band_structure.set_array('q_path', q_path)
    band_structure.set_array('frequencies', frequencies)
    band_structure.set_array('labels', band_labels)

    return {'band_structure': band_structure}


@make_inline
def create_volumes_inline(**kwargs):
    import numpy as np
    initial_structure = kwargs['structure']
    volume_relations = kwargs['volumes'].get_dict()['relations']

    structures = {}
    for i, vol in enumerate(volume_relations):
        cell = np.array(initial_structure.cell) * vol
        structure = StructureData(cell=cell)
        for site in initial_structure.sites:
            structure.append_atom(position=np.array(site.position) * vol, symbols=site.kind_name)
        structures["structure_{}".format(i)] = structure

    return structures


class WorkflowGruneisen(Workflow):
    def __init__(self, **kwargs):
        super(WorkflowGruneisen, self).__init__(**kwargs)

    # Calculates the reference crystal structure (optimize it if requested)
    @Workflow.step
    def start(self):
        self.append_to_report('Starting workflow_workflow')
        self.append_to_report('Phonon calculation of base structure')

        wf_parameters = self.get_parameters()
        # self.append_to_report('crystal: ' + wf_parameters['structure'].get_formula())

        wf_param_opt = dict(wf_parameters)

        wf = WorkflowPhonon(params=wf_param_opt)
        wf.store()

        #        wf = load_workflow(30)
        self.attach_workflow(wf)
        wf.start()

        self.next(self.volume_expansions)

    # Generate the volume expanded cells
    @Workflow.step
    def volume_expansions(self):
        self.append_to_report('Volume expansion calculations')
        wf_parameters = self.get_parameters()
        #   structure = wf_parameters['structure']

        structure = self.get_step(self.start).get_sub_workflows()[0].get_result('final_structure')

        inline_params = {'structure': structure,
                         'volumes': ParameterData(dict={ 'relations': [ 1.01, 0.99]})}  # plus, minus

        cells = create_volumes_inline(**inline_params)[1]


        for i, structures in enumerate(cells.iterkeys()):
            structure_vol = cells['structure_{}'.format(i)]
            self.append_to_report('structure_{}: {}'.format(i, structure_vol.pk))
            wf_parameters_volume = dict(wf_parameters)
            wf_parameters_volume['structure'] = structure_vol

            # Submit workflow
            wf = WorkflowPhonon(params=wf_parameters_volume, optimize=False)
            wf.store()

            self.attach_workflow(wf)
            wf.start()

        self.next(self.collect_data)

    # Collects the forces and prepares force constants
    @Workflow.step
    def collect_data(self):

        parameters = self.get_parameters()

        wf_origin= self.get_step(self.start).get_sub_workflows()[0]
        wf_plus, wf_minus = self.get_step(self.volume_expansions).get_sub_workflows()

        self.append_to_report('reading structure')

        inline_params = {'structure_origin': wf_origin.get_result('force_constants'),
                         'structure_plus':   wf_plus.get_result('force_constants'),
                         'structure_minus':  wf_minus.get_result('force_constants'),
                         'force_constants_origin': wf_origin.get_result('final_structure'),
                         'force_constants_plus':   wf_plus.get_result('final_structure'),
                         'force_constants_minus':  wf_minus.get_result('final_structure'),
                         'phonopy_input': parameters['phonopy_input']}

        # Do the phonopy Gruneisen parameters calculation
        results = phonopy_gruneisen_inline(**inline_params)[1]

        self.add_result('band_structure', results['band_structure'])

        self.append_to_report('Finishing workflow_workflow')

        self.next(self.exit)



