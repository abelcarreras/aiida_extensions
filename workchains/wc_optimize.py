# Works run by the daemon (using submit)

from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida.work.workchain import WorkChain, ToContext

from aiida.orm.data.base import Str, Float, Bool
from aiida.work.workchain import _If, _While

import numpy as np
from generate_inputs import *

class OptimizeStructure(WorkChain):
    """
    Workflow to calculate the force constants and phonon properties using phonopy
    """

    @classmethod
    def define(cls, spec):
        super(OptimizeStructure, cls).define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("machine", valid_type=ParameterData)
        spec.input("es_settings", valid_type=ParameterData)
        # Optional
        spec.input("pressure", valid_type=Float, required=False, default=Float(0.0))
        spec.input("tolerance_forces", valid_type=Float, required=False, default=Float(1e-5))
        spec.input("tolerance_stress", valid_type=Float, required=False, default=Float(1e-2))

        spec.outline(cls.optimize_cycle, _While(cls.not_converged)(cls.optimize_cycle), cls.get_data)

        #spec.outline(cls.optimize_cycle, cls.get_data)

    def not_converged(self):

        print ('Check convergence')
        self.report('tolerace  F:{} S:{}'.format(self.inputs.tolerance_forces, self.inputs.tolerance_stress))

        output_array = self.ctx.get('optimize').out.output_array
        forces = output_array.get_array('forces')
        stresses = output_array.get_array('stress')
        if len(stresses.shape) > 2:
            stresses = stresses[-1] * 10

        not_converged_forces = len(np.where(abs(forces) > float(self.inputs.tolerance_forces))[0])
        self.report('forces {}'.format(not_converged_forces))
        self.report(forces)

        stress_compare_matrix = stresses - np.diag([float(self.inputs.pressure)]*3)
        not_converged_stress = len(np.where(abs(stress_compare_matrix) > float(self.inputs.tolerance_stress))[0])

        self.report('stresses {}'.format(not_converged_stress))
        self.report(stresses)

        not_converged = not_converged_forces + not_converged_stress

        if not_converged == 0:
            print ('Converged')
            self.report('converged')
            return False

        print ('Not converged: {}'.format(not_converged))
        self.report('Not converged: {}'.format(not_converged))

        return True

    def optimize_cycle(self):

        # self.ctx._get_dict()
        print 'start optimization'

        if not 'optimize' in self.ctx:
            structure = self.inputs.structure
        else:
            structure = self.ctx.optimize.out.output_structure

        JobCalculation, calculation_input = generate_inputs(structure,
                                                            self.inputs.machine,
                                                            self.inputs.es_settings,
                                                            pressure=self.inputs.pressure,
                                                            type='optimize',
                                                            )

        # calculation_input._label = 'optimize'
        future = submit(JobCalculation, **calculation_input)
        print ('optimize calculation pk = {}'.format(future.pid))
        self.report('optimize calculation pk = {}'.format(future.pid))

        return ToContext(optimize=future)

    def get_data(self):
        print 'get_job'

        # self.ctx.structure = self.ctx.get('optimize').out.output_structure

        self.out('optimized_structure', self.ctx.optimize.out.output_structure)
        self.out('optimized_structure_data', self.ctx.optimize.out.output_parameters)

