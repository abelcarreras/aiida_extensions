.. AiiDA VASP Plugin documentation master file, created by
   sphinx-quickstart on Wed Feb 17 11:53:17 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AiiDA VASP Plugin's Documentation
============================================

AiiDA VASP Plugin is an AiiDA_ extension, writen in Python, which provides a user friendly and flexible way to run highly automated density functional theory (DFT) calculations using the Viena Ab initio Simulation Package, VASP_ . The code consists of two main parts, the Input and the Output plugin, which facilitate VASP calculation setup and output postprocessing within the AiiDA framework.

.. warning:: **This plugin is still in its early testing phase !!!**


Introduction
============
Covers a brief overview of the main functionality provided by the plugin.

.. toctree::
	introduction/intro

.. todo:: simplified input example, illustrate parser instructions !!!
.. todo:: output example

Installation
============
A quick guide to get you running.

.. toctree::
	install/dependencies
	install/plugin

Users
=====
An example based introduction to the code usage. Covers the plugin logic and the basics needed to run the VASP code.

.. toctree::
	examples/basic_usage.rst

.. Explain the scope of this part of the documentation --> objectives
.. Give a few use case examples to demonstrate the concepts.

Default Behaviour
=================
An overview of the features that may confuse you.

.. toctree::
	default_behaviour/defaults.rst


Advanced Users
==============
Extends the topic of Parser Instructions and gives an example of a new parser instruction implementation. 

.. note:: Knowledge of Python programing is assumed.

.. toctree::
	advanced/instruct

Developers
==========
An in-depth cover of the plugin implementation. 

.. note:: Advanced Python programing skills and familiarity with the AiiDA internals are assumed.

*TODO !*

.. toctree::
	devel/start


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: references.txt

