==============
How to install
==============


plugins
=======

copy (or link) the contents of plugins/jobs directory (including subdirectories) into **AIIDA_HOME**/aiida/orm/calculation/job/.
copy (or link) the contents of plugins/parsers directory into **AIIDA_HOME**/aiida/parsers/plugins/.

workflows
=========

copy (or link) the individual files in workflows directory (not including subfolders) into **AIIDA_HOME**/aiida/workflows/.

Final notes
===========

After installation, it may be necessary to restart the aiida daemon ::

    $ verdi daemon restart


