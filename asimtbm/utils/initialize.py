from activitysim.core import inject
from activitysim.core import tracing

from activitysim.core.steps.output import write_data_dictionary
from activitysim.core.steps.output import write_tables

from asimtbm.tables import zones

import logging
import warnings
import sys


def config_logger():
    """ActivitySim logger
    """
    tracing.config_logger()
    logging.captureWarnings(capture=True)
    warnings.simplefilter("always")

    logger = logging.getLogger('asimtbm')
    logger.info("Setup logger")


@inject.injectable(cache=True, override=True)
def trace_od(settings):
    return settings.get('trace_od', None)


@inject.injectable(cache=True)
def preload_injectables():
    inject.add_step('write_data_dictionary', write_data_dictionary)
    inject.add_step('write_tables', write_tables)
