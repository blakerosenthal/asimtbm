import logging
from os import path

import pandas as pd
import numpy as np

from activitysim.core import (
    inject,
    config,
    assign,
    pipeline,
    tracing,
)

from asimtbm.utils import skims
from asimtbm.utils import trips
from asimtbm.utils import tracing as trace

logger = logging.getLogger(__name__)

YAML_FILENAME = 'destination_choice.yaml'
ORIGIN_TRIPS_KEY = 'orig_zone_trips'


@inject.step()
def destination_choice(zones, data_dir, trace_od):
    """ActivitySim step that creates a raw destination choice table
    and calculates utilities.

    settings.yaml must specify 'destination_choice' under 'models' for
    this step to run in the pipeline. destination_choice.yaml must also
    specify the following:

        - spec_file_name: <expressions csv>
        - aggregate_od_matrices:
            - skims: <skims omx file>
        - dest_zone: <list of destination zone attribute columns>
        - orig_zone_trips: <dict of num trips for each segment>

    @inject.step before the method definition registers this step with the pipeline.

    Parameters
    ----------
    zones : pandas DataFrame of zone attributes
    data_dir :  str, data directory path
    trace_od : list or dict
        origin-destination pair to trace.

    Returns
    -------
    None
        but writes final dataframe to csv and registers it
        to the pipeline.
    """
    logger.info('running destination choice step ...')

    model_settings = config.read_model_settings(YAML_FILENAME)
    locals_dict = create_locals_dict(model_settings)

    od_index = create_od_index(zones.to_frame())

    skims_dict = skims.read_skims(zones.index, data_dir, model_settings)
    locals_dict.update(skims_dict)

    zone_matrices = create_zone_matrices(zones.to_frame(), od_index, model_settings)
    locals_dict.update(zone_matrices)

    segments = model_settings.get(ORIGIN_TRIPS_KEY)
    spec = read_spec_file(model_settings, segments)

    od_table = create_od_table(od_index, spec, locals_dict, trace_od)
    trips.calculate_num_trips(od_table, zones, spec, locals_dict,
                              segments, trace_od=trace_od)

    # This step is not strictly necessary since the pipeline
    # closes remaining open files on exit. This just closes them
    # now instead of leaving them open consuming memory for subsequent steps.
    skims.close_skims(locals_dict)

    logger.info('finished destination choice step.')


def create_locals_dict(model_settings):
    """Initial local parameters for the destination choice step.
    These will be expanded later and used in subsequent evaluations.

    Gets both constants and math expressions from model settings

    Parameters
    ----------
    model_settings : dict

    Returns
    -------
    dict
    """
    locals_dict = {}
    constants = config.get_model_constants(model_settings)
    math = model_settings.get('numpy')
    math_functions = {name: getattr(np, name) for name in math}
    locals_dict.update(constants)
    locals_dict.update(math_functions)

    return locals_dict


def create_od_index(zones_df):
    orig = np.repeat(np.asanyarray(zones_df.index), zones_df.shape[0])
    dest = np.tile(np.asanyarray(zones_df.index), zones_df.shape[0])
    od_df = pd.DataFrame({'orig': orig, 'dest': dest})

    return pd.MultiIndex.from_frame(od_df)


def read_spec_file(model_settings, segments):
    """Read expressions file from a csv using instructions from model settings.

    Parameters
    ----------
    model_settings: dict, from yaml
    segments : dict, origin zone trip segments

    Returns
    -------
    pandas DataFrame
    """
    spec_file_name = model_settings.get('spec_file_name')
    spec_file_path = config.config_file_path(spec_file_name, mandatory=True)

    logger.info('reading spec file \'%s\'' % spec_file_name)
    cfg = pd.read_csv(spec_file_path, comment='#')

    expected_header = ['description', 'target', 'expression', *segments.keys()]
    if not sorted(cfg.columns.values) == sorted(expected_header):
        raise RuntimeError("Spec file requires header %s" % expected_header)

    cfg.target = cfg.target.str.strip()
    cfg.expression = cfg.expression.str.strip()

    return cfg


def create_zone_matrices(zones, od_index, model_settings):
    """Convert zone values to flattened arrays

    Parameters
    ----------
    zones : pandas DataFrame
    od_index : pandas MultiIndex
    model_settings : dict

    Returns
    -------
    dictionary of flattened dest/orig zones
    """

    logger.info('creating zone matrices ...')

    dest_zones = zones[model_settings.get('dest_zone', [])]
    dest_zones.index.name = od_index.names[1]

    orig_zones = zones[model_settings.get('orig_zone', [])]
    orig_zones.index.name = od_index.names[0]

    return {
        'dest_zone': dest_zones.join(od_index.to_frame(), how='right'),
        'orig_zone': orig_zones.join(od_index.to_frame(), how='right')
    }


def create_od_table(od_index, spec, locals_dict, trace_od):
    """Assign variables with ActivitySim's assign and register output to pipeline

    Parameters
    ----------
    od_index : pandas MultiIndex
    spec : pandas DataFrame, assignment expressions
    locals_dict : dict,
        dictionary containing constants and zone matrices
    trace_od : list or dict, origin-destination pair

    Returns
    -------
    od_table : pandas DataFrame
        all origin-destination pairs
    """

    logger.info('creating OD table ...')

    od_df = od_index.to_frame(index=False)
    trace_rows = trace.trace_filter(od_df, trace_od)
    od_table, trace_results, _ = assign.assign_variables(spec, od_df,
                                                         locals_dict=locals_dict,
                                                         trace_rows=trace_rows)

    if trace_results is not None:
        tracing.write_csv(trace_results, file_name='od_table', transpose=False)

    od_table.set_index(od_index, inplace=True)

    logger.info('registering OD table to pipeline ...')
    pipeline.replace_table('od_table', od_table)
    create_zone_summary(od_table.reset_index())

    return od_table


def create_zone_summary(od_table):
    """Creates summary of od_table summed by origin zone

    Parameters
    ----------
    od_table : pandas DataFrame
        results of expression assignment

    Returns
    -------
    None
        registers summary to pipeline
    """
    logger.info('creating zone summary table ...')
    zone_summary = od_table.groupby(['orig']).sum()
    del zone_summary['dest']
    pipeline.replace_table('zone_summary', zone_summary)
