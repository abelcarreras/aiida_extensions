Installing the Plugin
=====================

The plugin consists of two parts: the input and the output plugin. These can be found in correspondingly named directories at the top-level of the AiiDA VASP Plugin distribution directory.

There are two ways to add the VASP Plugin to AiiDA.
You can either *symlink* or *copy* the plugin directories to the appropriate AiiDA distribution directory. The **directories must be renamed** as explained below. The package imports inside the modules will be broken if this is not done correctly!

.. note:: In the following we will assume that the AiiDA VASP Plugin is located inside the *~/aiida_vasp_plugin/* directory and the AiiDA distribution is located inside the *~/aiida_dir/*. **Please modify these paths correspondingly!**

To proceed with the installation follow method A or method B, according to your preferences.

.. highlight:: python

Method A - Symlink
------------------	
You need to do the following symlink:

.. code-block:: bash

	ln -s ~/aiida_vasp_plugin/input ~/aiida_dir/aiida/orm/calculation/job/vasp

for the input plugin, and:

.. code-block:: bash

	ln -s ~/aiida_vasp_plugin/output ~/aiida_dir/aiida/parsers/plugins/vasp

for the output plugin. 

If everything went ok, now you whould be able to import these modules from your python console.


Method B - Copy
---------------
First you need to copy the input plugin:

.. code-block:: bash

	cp -r ~/aiida_vasp_plugin/input ~/aiida_dir/aiida/orm/calculation/job/vasp

and then the output plugin:

.. code-block:: bash

	cp -r ~/aiida_vasp_plugin/output ~/aiida_dir/aiida/parsers/plugins/vasp

to the corresponding AiiDA directories.

If everything went ok you should be able to import these modules from your python console.


Check installation
------------------
Open your python console and try to import the packages:

.. code-block:: python

	import aiida.orm.calculation.job.vasp	
	import aiida.parsers.plugins.vasp

.. note:: If you get an error here please check that you have provided the correct paths to the copy/symlink commands above.


.. include:: ../references.txt
