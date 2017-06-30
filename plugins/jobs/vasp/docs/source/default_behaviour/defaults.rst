.. _default_behaviour:

Plugin's Default Behaviour
==========================

Plugin's default behaviour includes every bit of plugin's behaviour which is not obvious to the user.
These are mostly intended as commodity features, but for a new user it may lead to confusion.

.. note:: 
	Here we try to document all "hidden" features of the plugin which may result with undesired outcome for the user.
	**If a feature is not well documented, then it should be treated as a bug in the documentation!**

Mandatory & Optional Input
==========================

Preparation of the input for an AiiDA calculation was described :ref:`here <prep_aiida_calc>`. The main thing to recall is that we use the `calc.use_method`'s to pass the input to the AiiDA calculation object. 
It was noted that we did not use the `calc.use_structure_extras` method in that example.
Therefore, the `structure_extras` is an *optional* input parameter. 
Currently this is the only optional input !

.. note:: **All inputs are mandatory, except the structure_extras !**

Parser Instructions
===================

A significant fraction of implemented default behaviours is related to parser instructions. The parser instructions are used to specify how the calculation output is to be parsed. It is reasonable to assume that some basic quantities like energy and magnetization, will be of interest to all users. Hence, it is convenient to define *default parsers*. Since we divide the output into three parts: *errors, data, and structure*, we have an assigned default parser for each of these data types. More on these output types can be found :ref:`here <parser_instr>`.

Instruction Specification
-------------------------

What may be unusual is that **we require at least one parser to be specified for each of the data types above**. This involves a number of default behaviours:

1) If nothing is specified, the output plugin will load a default parser for each of the data types.

2) If only some of the parsers are not provided, the output plugin will load a default parser for the missing data type.

3) *However*, if a provided instruction can not be loaded during the calculation submission, e.g. if the calculation does not conform to the specifications or there is a typo in the input, the input plugin will *raise an exception*.

.. note:: The plugin will forgive you if a required instruction is not specified, and load defaults for you, *but it will crash* if you specify an instruction that can not be loaded.

Dummy Instructions
++++++++++++++++++

*There are perfectly legitimate reasons not to parse the whole output, we only require the user to be explicit about it !*

The dummy instructins are introduced to allow the user to express that he does not want to parse something, e.g. the user decides not to parse the output structure because he is doing static calculations and he wants to save memory space. 

The output nodes are still created, but they are empty! *(This may change in future versions!)*

.. note:: **Dummy instructions allow the user to skip the parsing !** 

Static Instructions
+++++++++++++++++++

Static parser instructions require the developer to specify the files which are required for parsing. This is used to automatically fetch the required files for the output parser. *The user does not need to provide any additional input !*

.. note:: **Mandatory input files are automatically appended to the retrieve list !**

Instruction Execution
---------------------

The output parsing is concieved as a loop that iterates over the list of parser instructions and executes them one-by-one, in an arbitrary order.
What happens if an instruction crashes during the execution ?! There are two possible scenarios:

1) the instruction handled the crash on it's own and returned without raising an exception. In this case you should probably expect to see an error message in the error log for that instruction, e.g. in the `errors@TheCrashedInstruction` node, but this depends on the implementation of that particular instruction.

2) the instruction didn't handle the crash. In this case the error log inside the `errors@TheCrashedInstruction` node will probably not be complete an we will have an additional error inside the `parser_warnings` node, created by the output parser itself, giving extra information about the crash.

.. warning:: 
	**Always check for output parsing errors !** 
	(If an instruction crashes the output parser will just move to the next instruction.)
