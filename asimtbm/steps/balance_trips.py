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
DEST_TARGETS = 'dest_zone_trip_targets'
ORIG_TARGETS = 'orig_zone_trip_targets'


@inject.step()
def balance_trips(zones, trace_od):
    """Improve the match between destination zone trip totals
    (given by the DEST_TARGETS in the balance_trips config file)
    and the trip counts calculated during the destination choice step.

    The config file should contain the following parameters:

    dest_zone_trip_targets:
      total: <aggregate destination zone trip counts>

      OR

      <segment_1>: totals for segment 1 (optional)
      <segment_2>: totals for segment 2 (optional)
      <segment_3>: totals for segment 3 (optional)

    (These are optional)
    max_iterations: maximum number of iteration to pass to the balancer
    balance_closure: float precision to stop balancing totals
    input_table: path to CSV to use instead of trips table.

    The config file can also have an orig_zone_trip_targets to manually
    specify origin zone totals instead of using the logsums calculated by
    the destination choice step.

    Parameters
    ----------
    zones : DataFrameWrapper
        zone attributes
    trace_od : list or dict


    Returns
    -------
    Nothing. Balances trips table and writes trace tables
    """

    logger.info('running trip balancing step ...')

    model_settings = config.read_model_settings(YAML_FILENAME)

    trips_df = get_trips_df(model_settings)
    trace_rows = trace.trace_filter(trips_df, trace_od)
    tracing.write_csv(trips_df[trace_rows],
                      file_name='trips_unbalanced',
                      transpose=False)

    trips_df = trips_df.melt(
                id_vars=['orig', 'dest'],
                var_name='segment',
                value_name='trips')

    dest_targets = model_settings.get(DEST_TARGETS)
    orig_targets = model_settings.get(ORIG_TARGETS)
    max_iterations = model_settings.get('max_iterations', 50)
    closure = model_settings.get('balance_closure', 0.001)

    aggregates, dimensions = calculate_aggregates(trips_df,
                                                  zones.to_frame(),
                                                  dest_targets,
                                                  orig_targets)

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


def get_trips_df(model_settings):
    """Default to pipeline trips table unless
    user provides a CSV
    """
    filename = model_settings.get('input_table', None)

    if not filename:
        trips_df = pipeline.get_table('trips')
        return trips_df.reset_index()

    fpath = config.data_file_path(filename, mandatory=True)

    return pd.read_csv(fpath, header=0, comment='#')


def calculate_aggregates(df, zones, dest_targets, orig_targets=None):
    """Calculates grouped totals along specified dataframe dimensions

    Parameters
    ----------
    df : pandas DataFrame
    zones : DataFrame
    dest_targets : dict
        segment:vector pair where vector is target aggregate total
        for destination sums
    orig_targets : dict (optional)
        segment:vector pair where vector is target aggregate total
        for origin sums. Balances against existing origin sums if None.

    Returns
    -------
    aggregates : list of pandas groupby objects
    dimensions : list of lists of column names that match aggregates
    """

    targets = {
        'dest': dest_targets,
        'orig': orig_targets,
    }

    aggregates = []
    dimensions = []

    for level, target in targets.items():
        if not target:

            logger.info('no %s targets found. using existing table sums.' % level)

            sums = df.groupby([level, 'segment'])['trips'].sum()
            aggregates.append(sums)
            dimensions.append([level, 'segment'])

        elif 'total' in target:

            logger.info('using %s vector for aggregate %s target totals'
                        % (target['total'], level))

            target_vector = zones[target['total']]
            target_vector.index.name = level
            aggregates.append(target_vector)
            dimensions.append([level])

        else:

            logger.info('using %s vectors for aggregate %s target totals'
                        % (list(target.values()), level))

            target_df = zones[list(target.values())].copy()
            mapping = dict((v, k) for k, v in target.items())
            target_df = target_df.rename(columns=mapping)
            target_sums = target_df.stack()
            target_sums.index.names = [level, 'segment']
            target_sums.name = 'trips'
            aggregates.append(target_sums)
            dimensions.append([level, 'segment'])

    return aggregates, dimensions
