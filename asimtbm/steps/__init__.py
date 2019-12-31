from activitysim.core import inject as _inject

from activitysim.core.steps.output import write_data_dictionary
from activitysim.core.steps.output import write_tables

from . import balance_trips
from . import destination_choice


@_inject.injectable(cache=True)
def preload_injectables():
    _inject.add_step('write_data_dictionary', write_data_dictionary)
    _inject.add_step('write_tables', write_tables)
