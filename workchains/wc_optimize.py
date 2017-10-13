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

    def not_converged(self):
        return False

    def optimize_cycle(self):

        # self.ctx._get_dict()
        print 'start optimization'
        if not 'structure' in self.ctx:
            structure = self.inputs.structure
        print 'got structure'

        try:
            plugin = Code.get_from_string(self.inputs.es_settings.dict.code).get_attr('input_plugin')
            # plugin = self.inputs.es_settings.dict.code.get_attr('input_plugin')
        except:
            plugin = Code.get_from_string(self.inputs.es_settings.dict.code_forces).get_attr('input_plugin')
            # plugin = self.inputs.es_settings.dict.code_forces.get_attr('input_plugin')

        JobCalculation, calculation_input = generate_inputs(self.inputs.structure,
                                                            self.inputs.machine,
                                                            self.inputs.es_settings)

        print 'job created'
#        calculation_input._label = label
        future = submit(JobCalculation, **calculation_input)
        calcs = {'optimize': future}
        print 'job sent'
        return ToContext(**calcs)

    def get_data(self):
        print 'get_job'
        self.out('optimized_structure', self.ctx.get('optimize'))

