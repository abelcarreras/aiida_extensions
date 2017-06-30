.. _parser_instr:

Introduction to Parser Instructions
===================================

Parser instructions are the central concept of the output plugin.
They provide the output plugin with an easily configurable, and extensible, parsing functionality.
The role of the output plugin can thus be understood, in simple terms, as a boiler-plate needed to load, execute, and store the results returned by the parser instructions. 

.. note:: *Parsing of the output is achieved by executing a sequence of parser instructions!*

Specifying the Parser Instruction Input
---------------------------------------

In order to customize the output parsing process we need to specify which instructions should be used as a part of the input.
The instructions are specified using a special key `PARSER_INSTRUCTIONS`, within the `settings` input node, as shown below:

.. code-block:: python

	settings = {'PARSER_INSTRUCTIONS': []}
	instr = settings['PARSER_INSTRUCTIONS']  # for easier access
	instr.append({
	    'instr': 'dummy_data_parser',
	    'type': 'data',
	    'params': {}
	})

	...

	calc.use_settings(ParameterData(dict=settings))

Where the `calc` is an instance of the `VaspCalculation` class.

In the example above we are appending a single `data` parser instruction called `dummy_data_parser`. 
**The parser instructions are supposed to be specified as a dictionary with three keys:** `instr`, `type`, and `params`.

Currently there are three parser instruction types implemented: `data`, `error`, and `structure`. 
The distinction between these types comes into play during the instruction loading, where the output parser appends different auxiliary parameters to the instruction based on its type.
For example, to every `error` type instruction a `SCHED_ERROR_FILE` parameter is appended. 
More information about the plugin's default behaviour can be found :ref:`here <default_behaviour>`.	

Defining New Parser Instructions
--------------------------------

All parser instructions inherit from the base class, `BaseInstruction`, which provides the interface towards the output plugin.
Therefore, the `BaseInstruction` is *a template for implementing custom parser instructions*.

In order to implement a new parser instruction one must inherit from the base class and overrride the two following things:

1) list of input files, given by the class property `_input_file_list_`, or by setting the `_dynamic_file_list_ = True`, when the names of the input files are not known in advance.
2) override the `BaseInstruction._parser_function(self)`, which is a method that is called when the instruction is executed - it implements the actual parsing of the output.

Below we give examples on how to implement these two different instruction types.

.. note:: In future versions we may implement `BaseInstruction` subclasses for each instruction type, i.e. `StaticInstruction` and `DynamicInstruction`, in order to be explicit about our intents.

Static Parser Instruction
-------------------------

Static parser instruction is just an ordinary parser instruction for which we can specify the list of input file names in advance, i.e. the *input file names are static*.

The use of this method is advantageous since the input plugin will automatically update the list of files to be retreived, and the instruction itself will automatically check if the required files are present before the parsing starts. 

Since the **static parser instructions** offer both the user commodity and additional safeties against an invalid user input, they should be **prefered** over the dynamic parser instructions!

Example:
++++++++

The `Default_vasprun_parserInstruction` is an example of the static parsing instruction.
It operates only on the statically named `vasprun.xml` file.

First thing in defining a static parsing instruction is to override the `_input_file_list_`:

.. literalinclude:: ../../../output/instruction/data/default_vasprun_parser.py
   :lines: 22-30

In the case above the only input file is the `vasprun.xml`. 

Next follows the implementation of the `_parser_function`.
The `_parser_function` implements the output parsing logic. This part depends only on the user preferences and does not depend on the internal working of the AiiDA.

Finally, **the output** must be returned as a tuple:

.. literalinclude:: ../../../output/instruction/data/default_vasprun_parser.py
   :lines: 142

The `nodes_list` is just an arbitrary *list of tuples*, e.g. `[('velocities', ArrayData_type), ('energies', ParameterData_type), ...]`, where the second tuple value needs to be an instance of the AiiDA's `Data` type.

The second item in the return tuple is the `parameter_warnings` object, which is just a dictionary in which we can log useful information, e.g. non-critical errors, during the instruction execution. For example:

.. code-block:: python

	parser_warnings.setdefault('error name', 'details about the error')

After the instruction returns, the parser warnings are converted to a node, `('errors@Default_vasprun_parserInstruction', ParameterData(parser_warnings))`, which is stored in the AiiDA database as a part of the output.

.. note:: In summary, static parser instruction is implemented by overriding the `_input_file_list_` and the `_parser_function`. The parsed output must be returned in a format described above.

Dynamic Parser Instruction
--------------------------

Dynamic parser instruction differ from the static parser instructions in that the *input file names must be provided by the user* as an instruction parameter during the instruction specification, i.e. during the VASP calculation setup. This represents and overhead and allows for a typo to cause an instruction execution crash during the output parsing. For this reason the static parser methods should be used whenever that is possible.

Example:
++++++++

The `Default_error_parserInstruction` is an example of the dynamic parser instruction.

The whole code is given below:

.. literalinclude:: ../../../output/instruction/error/default_error_parser.py
	:lines: 18-52

First the `_dynamic_files_list_` is set to `True`, followed by the `_parser_function` implementation:

1. get the input file name, `self._params['SCHED_ERROR_FILE']`, to open for parsing. (See the note below.)

2. read the whole standard error file. We could be looking for a particular kind of error here instead!

3. set up the output node list and return. In this example only one node, `runtime_errors`, is created. The `parser_warnings` is just an empty dictionary.


.. note:: The `SCHED_ERROR_FILE` parameter is appended automatically by the output parser to every `error` instruction type. This is an example of the :ref:`default behaviour <default_behaviour>`.
