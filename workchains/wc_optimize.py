# Works run by the daemon (using submit)

from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida.work.workchain import WorkChain, ToContext

from aiida.orm import Code, CalculationFactory
from aiida.orm.data.base import Str, Float, Bool
from aiida.work.workchain import _If, _While


PwCalculation = CalculationFactory('quantumespresso.pw')
PhonopyCalculation = CalculationFactory('phonopy')

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
        # Should be optional
        spec.input("pressure", valid_type=Float)

        spec.outline(cls.optimize_cycle, _While(cls.not_converged)(cls.optimize_cycle), cls.get_data)

        #spec.outline(cls.optimize_cycle, cls.get_data)

    def not_converged(self):
        tolerance_forces = 1e-5
        tolerance_stress = 1e-2

        print ('Check convergence')

        if not 'presure' in self.inputs:
            pressure = 0

        output_array = self.ctx.get('optimize').out.output_array
        forces = output_array.get_array('forces')
        stresses = output_array.get_array('stress')

        not_converged_forces = len(np.where(abs(forces) > tolerance_forces)[0])
        if len(stresses.shape) > 2:
            stresses = stresses[-1] * 10

        not_converged_stress = len(np.where(abs(np.diag(stresses) - pressure) > tolerance_stress)[0])
        np.fill_diagonal(stresses, 0.0)
        not_converged_stress += len(np.where(abs(stresses) > tolerance_stress)[0])
        not_converged = not_converged_forces + not_converged_stress

        if not_converged == 0:
            print ('Converged')
            return False

        print ('Not converged: {}'.format(not_converged))
        return False

    def optimize_cycle(self):

        # self.ctx._get_dict()
        print 'start optimization'
        if not 'structure' in self.ctx:
            structure = self.inputs.structure
        print 'got structure'

        JobCalculation, calculation_input = generate_inputs(structure,
                                                            self.inputs.machine,
                                                            self.inputs.es_settings,
                                                            type='optimize'
                                                            )

        # calculation_input._label = 'optimize'
        future = submit(JobCalculation, **calculation_input)
        print ('pk = {}'.format(future.pid))
        calcs = {'optimize': future}
        print 'job sent'
        return ToContext(**calcs)

    def get_data(self):
        print 'get_job'

        self.ctx.structure = self.ctx.get('optimize').out.output_structure

        self.out('optimized_structure', self.ctx.get('optimize').out.output_structure)
        self.out('optimized_structure_data', self.ctx.get('optimize').out.output_parameters)

