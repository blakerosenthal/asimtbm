import logging
import pandas as pd

from asimtbm.utils.matrix_balancer import Balancer

from activitysim.core import (
    inject,
    config,
    tracing,
    pipeline
)

from asimtbm.utils import tracing as trace

logger = logging.getLogger(__name__)

YAML_FILENAME = 'balance_trips.yaml'
TARGETS_KEY = 'dest_zone_trip_targets'


@inject.step()
def balance_trips(trips, zones, trace_od):
    """Improve the match between destination zone trip totals
    (given by the TARGETS_KEY in the balance_trips config file)
    and the trip counts calculated during the destination choice step.

    The config file should contain the following parameters:

    dest_zone_trip_targets:
      total: <aggregate destination zone trip counts>
      <segment_1>: totals for segment 1 (optional)
      <segment_2>: totals for segment 2 (optional)
      <segment_3>: totals for segment 3 (optional)

    (These are optional)
    max_iterations: maximum number of iteration to pass to the balancer
    balance_closure: float precision to stop balancing totals

    Parameters
    ----------
    trips : DataFrameWrapper
        OD trip counts
    zones : DataFrameWrapper
        zone attributes
    trace_od : list or dict


    Returns
    -------
    Nothing. Balances trips table and writes trace tables
    """

    logger.info('running trip balancing step ...')

    model_settings = config.read_model_settings(YAML_FILENAME)
    targets = model_settings.get(TARGETS_KEY)

    trips_df = trips.to_frame().reset_index()
    trace_rows = trace.trace_filter(trips_df, trace_od)
    tracing.write_csv(trips_df[trace_rows],
                      file_name='trips_unbalanced',
                      transpose=False)

    trips_df = trips_df.melt(
                id_vars=['orig', 'dest'],
                var_name='segment',
                value_name='trips')

    max_iterations = model_settings.get('max_iterations', 50)
    closure = model_settings.get('balance_closure', 0.001)

    aggregates, dimensions = calculate_aggregates(trips_df, zones.to_frame(), targets)

    balancer = Balancer(trips_df.reset_index(),
                        aggregates,
                        dimensions,
                        weight_col='trips',
                        max_iteration=max_iterations,
                        closure=closure)
    balanced_df = balancer.balance()

    balanced_trips = balanced_df.set_index(['orig', 'dest', 'segment'])['trips'].unstack()
    tracing.write_csv(balanced_trips.reset_index()[trace_rows],
                      file_name='trips_balanced',
                      transpose=False)
    pipeline.replace_table('trips', balanced_trips)

    logger.info('finished balancing trips.')


def calculate_aggregates(df, zones, targets):
    """Calculates grouped totals along specified dataframe dimensions

    Parameters
    ----------
    df : pandas DataFrame
    zones : DataFrame
    targets : dict
        segment:vector pair where vector is target aggregate total

    Returns
    -------
    aggregates : list of pandas groupby objects
    dimensions : list of lists of column names that match aggregates
    """

    # must preserve origin totals calculated by dest_choice step
    orig_sums = df.groupby(['orig', 'segment'])['trips'].sum()
    aggregates = [orig_sums]
    dimensions = [['orig', 'segment']]

    if 'total' in targets:

        logger.info('using %s vector for aggregate destination target totals'
                    % targets['total'])

        dest_targets = zones[targets['total']]
        dest_targets.index.name = 'dest'
        aggregates.append(dest_targets)
        dimensions.append(['dest'])

    else:

        logger.info('using %s vectors for aggregate destination target totals'
                    % list(targets.values()))

        dest_df = zones[list(targets.values())].copy()
        mapping = dict((v,k) for k,v in targets.items())
        dest_df = dest_df.rename(columns=mapping)
        dest_sums = dest_df.stack()
        dest_sums.index.names = ['dest', 'segment']
        dest_sums.name = 'trips'
        aggregates.append(dest_sums)
        dimensions.append(['dest', 'segment'])


    return aggregates, dimensions
