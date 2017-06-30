AiiDA VASP Plugin Introduction
==============================

The AiiDA VASP Plugin aims to provide a full support for running VASP_ calculations using the AiiDA_ package.

The User Interface is based heavily on the VASP support already provided by the `Materials Project`_ through pymatgen_.
Main reasons for this are:
	* to support a **standard VASP interface** across different high-throughput frameworks.
	* to avoid code duplication and utilize the mature and well supported pymatgen code base.


The *philosophy* behind this plugin can be summarised as follows: 
	*Take a well defined VASP input and return parsed data in a form the User has specified - no more, no less.*

In order to facilitate the latter, we have implemented **Parser Instructions**, a simple way to customize and extend the parsing capabilities of the output parser.


.. _Materials Project: http://www.materialsproject.org/
.. include:: ../references.txt
