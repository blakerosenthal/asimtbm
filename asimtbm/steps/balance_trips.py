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

    dest_targets = zones[targets]
    dest_targets.index.name = 'dest'

    segment_sums = trips_df.groupby(['orig', 'segment'])['trips'].sum()

    aggregates = [
        dest_targets,
        segment_sums,
    ]

    dimensions = [['dest'], ['orig', 'segment']]
    max_iterations = model_settings.get('max_iterations', 50)
    closure = model_settings.get('balance_closure', 0.001)

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
