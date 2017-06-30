Basic Usage
===========

In this section we cover the basics of setting up a VASP calculation using the plugin.
The procedure can be logically split into two steps. The first step is to set up VASP input using the VASP interface provided by the pymatgen_ package. In the second step these objects, together with a set of user specified :ref:`output parser instructions <parser_instr>`, are passed as an input to the AiiDA calculation.

.. note:: 
	**The pymatgen syntax will not be covered here in great detail!** - just a short use-case example will be provided. 
	For more details on pymatgen we refer you to `pymatgen documentation <pymatgen>`_.


Preparing Pymatgen Input
========================

A short example of setting up pymatgen VASP input is given below. The goal is to create: `POSCAR`, `INPUTCAR`, `KPOINTS`, and `POTCAR` files, which represent a minimal input for any VASP calculation.

An excerpt from the full code is shown below to illustrate the input setup procedure:

.. literalinclude:: ./SubmittingJob.py
	:lines: 4-9, 48-64, 69-72

Therefore, for each VASP input file we have a pymatgen object representing it, e.g. `KPOINTS` is represented by the `pymatgen.io.vasp.Kpoints` object. Our task here is just to provide basic information needed to construct the VASP input files. 

**Full code** used for this example can be found :ref:`here <full_pmg_input>` . 

.. _prep_aiida_calc:

Preparing AiiDA calculation
===========================

The aim of this section is to set up a working AiiDA calculation.
We will assume that all pymatgen objects representing the VASP input have already been created. Our task then is to create a VASP calculation object and pass it the content of the pymatgen input files.

Before we pass the input files to the AiiDA calculation we need to **split** the `POSCAR` file, since it may contain both dictionary and array data. This is achieved by the `disassemble_poscar` function which returns a dictonary of `POSCAR` parts. It its important to note that each of these parts is already an instance of AiiDA's `Data` class and can be directly stored in the AiiDA database. The split is done like this:

.. literalinclude:: ./SubmittingJob.py
	:lines: 10-15, 74-75

.. note:: This intermediate step represents only a transitional solution which will be improved in future versions! 

The next step is to create an instance of the AiiDA VASP calculation and pass it the input files. The code to do this is shown below:

.. literalinclude:: ./SubmittingJob.py
	:lines: 74-79, 82-127

The calculation can now be submitted.

What is **important to notice** are the `calc.use_method`'s which are specific to the VASP plugin.
These can be logically divided into four groups:

	* *use_incar, use_potcar, use_kpoints* - passed as a `ParameterData` object, which store the `dict` representation of the pymatgen object
	* *use_poscar, use_structure, use_structure_extras* - passed as correspondingly named objects in the `poscar_parts` dict, which was obtained 		  by splitting up the `POTCAR` object. **Note:** the `structure_extras` in the example is not shown because this data is optional, i.e. it 		  may contain array data that can be found in the `CONTCAR` file, e.g. the final velicities of ions, etc.
	* *use_settings* - pased as `ParameterData`. Used to specify additional files to retreive and output parser instructions.
	* *use_chgcar*, *use_wavecar* - passed as a `SinglefileData` object. See the next section for more details on using these inputs.


**Full code** used for this example can be found :ref:`here <full_aiida_input>` .

CHGCAR and WAVECAR Files
------------------------

The `CHGCAR` and `WAVECAR` files are usually used for continuation runs.
The plugin treats them as an *optional input*.
The `SinglefileData` object can be created like this:

.. code-block:: python

	from aiida.orm.data.singlefile import SinglefileData

	input_file = SinglefileData()
	input_file.set_file('path/to/the/file/CHGCAR')

The `input_file` now points to the actual file on the disc and will be copied to the AiiDA database when the calculation's `store_all` method is called.
It is important to note here that we **must** have an input `CHACAR/WAVECAR` file written at some location on the disc before we can create a `SinglefileData` object.

Once we have created a `SinglefileData` representation of the `CHACAR/WAVECAR` file we can pass it to AiiDA as an input like this:

.. code-block:: python
	
	chgcar = SinglefileData()
	chgcar.set_file('path/to/the/file/CHGCAR')
	...
	calc.use_chgcar(chgcar)

and similarly for the `WAVECAR` file.

.. include:: ../../references.txt
