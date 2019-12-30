from activitysim.core import inject as _inject
from activitysim.core import tracing as _tracing

import logging as _logging
import warnings as _warnings


def config_logger():
    """ActivitySim logger
    """
    _tracing.config_logger()
    _logging.captureWarnings(capture=True)
    _warnings.simplefilter("always")

    logger = _logging.getLogger('asimtbm')
    logger.info("Setup logger")


@_inject.injectable(cache=True, override=True)
def trace_od(settings):
    return settings.get('trace_od', None)
