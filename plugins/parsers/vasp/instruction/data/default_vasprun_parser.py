# -*- coding: utf-8 -*-

# imports here
from aiida.parsers.exceptions import OutputParsingError
#
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
#
from aiida.parsers.plugins.vasp.instruction import BaseInstruction
#
from pymatgen.io import vasp


__copyright__ = u'Copyright © 2016, Mario Zic, Trinity College Dublin. All Rights Reserved.'
__license__ = "Apache, Version 2.0, see LICENSE.txt file"
__version__ = "0.0.1"
__contributors__ = "Mario Zic"
__contact__ = u'zicm_at_tcd.ie'

# This file has been modified respect to the original code by Mario Zic

class Default_vasprun_parserInstruction(BaseInstruction):

    _input_file_list_ = ['vasprun.xml']

    def _parser_function(self):
        """
        Parses the vasprun.xml using the Pymatgen Vasprun function.
        """

        vasp_param = {}  # ParameterData
        vasp_array = {}  # ArrayData

        parser_warnings = {}  # return non-critical errors

        # parameter data keys
        _vasprun_keys = [  # directly accessible, by name
            # logical
            'converged',
            'converged_electronic',
            'converged_ionic',
            'dos_has_errors',
            'is_spin',
            'is_hubbard',
            # value
            'efermi'
        ]
        _vasprun_special_keys = [  # can't be accessed directly
            # logical
            'is_band_gap_direct',
            # value
            'no_ionic_steps',
            'final_energy',  # returned as FloatWithUnit
            'energy_units',
            'free_energy',
            'energy_wo_entropy',
            'energy_T0',
            'entropy_TS',
            'no_electronic_steps',
            'total_no_electronic_steps',
            'band_gap',
            'cbm',
            'vbm',
        ]

        # array data keys
        # TODO
        # list --> array; see what exactly ArrayData supports

        # parsing output files
        try:
            vspr = vasp.Vasprun(self._out_folder.get_abs_path('vasprun.xml'))
        except Exception, e:
            msg = (
                "Parsing vasprun file in pymatgen failed, "
                "with error message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        # extract data
        try:
            for k in _vasprun_keys:
                vasp_param[k] = getattr(vspr, k)

            # acessing special keys manually
            band_gap, cbm, vbm, is_direct = tuple(
                vspr.eigenvalue_band_properties)
            vasp_param['band_gap'] = band_gap
            vasp_param['cbm'] = cbm
            vasp_param['vbm'] = vbm
            vasp_param['is_band_gap_direct'] = is_direct

            vasp_param['no_ionic_steps'] = len(vspr.ionic_steps)

            last_ionic = vspr.ionic_steps[-1]
            vasp_param['free_energy'] = last_ionic['e_fr_energy']
            vasp_param['energy_wo_entropy'] = last_ionic['e_wo_entrp'] # CHECK: looks like a bug in pymatgen !!!
            vasp_param['energy_T0'] = last_ionic['e_0_energy']
            vasp_param['entropy_TS'] = (
                last_ionic['e_fr_energy'] - last_ionic['e_wo_entrp']
            )
            vasp_param['no_electronic_steps'] = len(
                last_ionic['electronic_steps'])
            vasp_param['total_no_electronic_steps'] = 0
            for step in vspr.ionic_steps:
                vasp_param['total_no_electronic_steps'] += len(
                    step['electronic_steps']
                )

            final_en = vspr.final_energy
            vasp_param['final_energy'] = final_en.real
            vasp_param['energy_units'] = str(final_en.unit)

        except Exception, e:
            msg = (
                "Processing of extracted data failed, "
                "with error message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

        # construct proper output format
        try:
            nodes_list = []
            parameter_data = ParameterData(dict=vasp_param)
            nodes_list.append((
                'vasp_parameters@{}'.format(self.__class__.__name__),
                parameter_data
            ))
        except Exception, e:
            msg = (
                "Failed to create AiiDA data structures "
                "(ParameterData/ArrrayData) from parsed data, "
                "with error message:\n>> {}".format(e)
            )
            raise OutputParsingError(msg)

#        parser_warnings.setdefault('test', 'dummy') # test

        if not parser_warnings:
            parser_warnings = None

        return nodes_list, parser_warnings
