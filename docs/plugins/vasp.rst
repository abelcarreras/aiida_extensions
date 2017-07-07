====
VASP
====

Vasp plugin requires some additional configuration.
Pseudopotential has to be set in pymatgen using the PMG command

pmg config -p pseudo_original_dir  pseudo_destination_dir
pmg config --add VASP_PSP_DIR pseudo_destination_dir

where pseudo_original_dir is the pseudopotentials directory where the pseudopotentials are
as vasp provides, and pseudo_destination_dr is the directory where the pseudopotentials will be
copied and arranged for Pymatgen to use.

Refere to pymatgen manual for more information