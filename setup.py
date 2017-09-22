from setuptools import setup, find_packages

setup(
    name='aiida-phonon',
    version='0.1',
    description='AiiDA plugin for running phonon calculations using phonopy',
    url='https://github.com/abelcarreras/aiida_extensions',
    author='Abel Carreras',
    author_email='abelcarreras@gmail.com',
    license='MIT license',
    packages=find_packages(exclude=['aiida']),
    setup_requires=['reentry'],
    reentry_register=True,
    entry_points={
        'aiida.calculations': [
            'lammps.combinate = plugins.jobs.lammps.combinate:CombinateCalculation',
            'lammps.force = plugins.jobs.lammps.force:ForceCalculation',
            'lammps.md = plugins.jobs.lammps.md:MdCalculation',
            'lammps.optimize = plugins.jobs.lammps.optimize:OptimizeCalculation',
            'phonopy = plugins.jobs.phonopy: PhonopyCalculation',
            'dynaphopy = plugins.jobs.dynaphopy: DynaphopyCalculation'],
        'aiida.parsers': [
            'lammps.force = plugins.parsers.lammps.force:ForceParser',
            'lammps.md = plugins.parsers.lammps.md:MdParser',
            'lammps.optimize = plugins.parsers.lammps.optimize:OptimizeParser',
            'phonopy = plugins.parsers.phonopy: PhonopyParser',
            'dynaphopy = plugins.parsers.dynaphopy: DynaphopyParser'],
        'aiida.workflows': [
            'wf_phonon = workflows.wf_phonon:WorkflowPhonon',
            'wf_grunesien_pressure = workflows.wf_gruneisen_pressure:WorkflowGruneisen',
            'wf_grunesien_volume = workflows.wf_gruneisen_volume:WorkflowGruneisen',
            'wf_qha = workflows.qha:WorkflowQHA',
            'wf_quasiparticle = workflows.quasiparticle:WorkflowQuasiparticle',
            'wf_quasiparticle_thermo = workflows.wf_quasiparticle_thermo:WorkflowQuasiparticle',
            'wf_scan_quasiparticle = workflows.wf_scan_quasiparticle:WorkflowScanQuasiparticle',
        ]
    }
)

