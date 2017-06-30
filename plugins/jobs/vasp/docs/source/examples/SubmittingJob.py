# -*- coding: utf-8 -*-

# imports here
import numpy as np
#
# Pymatgen imports
import pymatgen as mg
from pymatgen.io import vasp as vaspio
#
# AiiDA imports
from aiida.orm import Code, DataFactory
from aiida.orm.calculation.job.vasp import vasp as vplugin
from aiida import load_dbenv
load_dbenv()

__copyright__ = u'Copyright © 2016, Mario Zic. All Rights Reserved.'
__contact__ = u'mario.zic.st_at_gmail.com'


# main body below

# === Prepare Input ===
# INCAR
incar_dict = {
    "NPAR": 24,
    "NELM": 2,
    "ISTART": 0,
    "ICHARG": 2,
    "MAGMOM": "5.0 -5.0 0.0",
    "IBRION": -1,
    "NSW": 0,
    "ISIF": 2,
    "NBANDS": 72,  # you may want to change this
    "ISPIND": 2,
    "ISPIN": 2,
    "ISYM": 1,
    "LWAVE": ".FALSE.",
    "LCHARG": ".TRUE.",
    "PREC": "Accurate",
    "ENCUT": 300,
    "EDIFF": 1e-06,
    "ALGO": "Fast",
    "ISMEAR": 1,
    "SIGMA": 0.05
}
incar = vaspio.Incar(incar_dict)

# POSCAR
lattice_constant = 5.97
lattice = lattice_constant * np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]
])
lattice = mg.Lattice(lattice)

struct = mg.Structure(
    lattice,
    [Mn, Mn, Ga],
    # site coords
    [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25], [0.50, 0.50, 0.50]]
)
poscar = vaspio.Poscar(struct, comment='cubic Mn2Ga')

# POTCAR
# Note: for this to work Pymatgen needs to have an access to VASP pseudopotential directory
potcar = vaspio.Potcar(symbols=['Mn_pv', 'Mn_pv', 'Ga_d'], functional='PBE')

# KPOINTS
kpoints = vaspio.Kpoints.monkhorst_automatic(
    kpts=(10, 10, 10), shift=(0.0, 0.0, 0.0)
)

# split the poscar for AiiDA serialization
poscar_parts = vplugin.disassemble_poscar(poscar)

# === Prepare Calculation
ParameterData = DataFactory('parameter')
StructureData = DataFactory('structure')

submit_test = True  # CAUTION: changing this will affect your database

codename = 'Vasp'  # this may be differ from user-to-user
code = Code.get(codename)  # executable to call, module imports etc

calc = code.new_calc()
calc.label = "VASP plugin development"
calc.description = "Test input plugin"
calc.set_max_wallclock_seconds(5*60)  # 5 min
calc.set_resources({
    "num_machines": 1,
    "num_mpiprocs_per_machine": 1,
    'num_cores_per_machine': 24  # this will differ from machine-to-machine
})
calc.set_withmpi(True)

calc.use_poscar(poscar_parts['poscar'])
calc.use_structure(poscar_parts['structure'])
calc.use_incar(
    ParameterData(dict=incar.as_dict())
)
calc.use_kpoints(
    ParameterData(dict=kpoints.as_dict())
)
calc.use_potcar(
    ParameterData(dict=potcar.as_dict())
)

# settings
settings = {'PARSER_INSTRUCTIONS': []}
pinstr = settings['PARSER_INSTRUCTIONS']
pinstr.append({
    'instr': 'dummy_data',
    'type': 'data',
    'params': {}
})

# additional files to return
settings.setdefault(
    'ADDITIONAL_RETRIEVE_LIST', [
        'OSZICAR',
        'CONTCAR',
        'OUTCAR',
        'vasprun.xml'
        ]
)
calc.use_settings(ParameterData(dict=settings))

# NOTE: you may need this line depending on your environment
#calc.set_custom_scheduler_commands(
#    """#PBS -A your_project_account_code_here
#    """
#)

if submit_test:
    subfolder, script_filename = calc.submit_test()
    print "Test_submit for calculation (uuid='{}')".format(calc.uuid)
    print "Submit file in {}".format(os.path.join(
        os.path.relpath(subfolder.abspath),
        script_filename
    ))
else:
    calc.store_all()
    print "created calculation; calc=Calculation(uuid='{}') # ID={}".format(
        calc.uuid, calc.dbnode.pk)
    calc.submit()
    print "submitted calculation; calc=Calculation(uuid='{}') # ID={}".format(
        calc.uuid, calc.dbnode.pk)
