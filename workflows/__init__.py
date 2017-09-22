from aiida.orm.workflow import Workflow

def PhononWorkflowsFactory(module):

    from aiida.common.pluginloader import BaseFactory
    return BaseFactory(module, Workflow, 'aiida.workflows.phonon')