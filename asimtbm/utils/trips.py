import logging
import pandas as pd
import numpy as np

from activitysim.core import assign, tracing, pipeline
from asimtbm.utils import tracing as trace

logger = logging.getLogger(__name__)


def calculate_num_trips(od_df, zones, spec, locals_dict, segments, trace_od=None):
    """Calculate number of trips for each origin-destination zone pair for
    each segment.

    Parameters
    ----------
    od_df : pandas DataFrame
        origin-destination DataFrame of target variables calculated
        from the expressions file
    zones : pandas DataFrame, zones table
    spec : pandas DataFrame, assignment expressions
    locals_dict : dict
        dictionary of constants, etc.
    segments : dict
        dictionary of segments. key is segment name, value is
        corresponding column in zones table
    trace_od : list or dict, origin-destination pair

    Returns
    -------
    None
        writes calculations trace and registers trips table to pipeline
    """
    logger.info('calculating number of trips per segment ...')
    trips_df = pd.DataFrame(index=od_df.index)
    trace_rows = trace.trace_filter(trips_df.reset_index(), trace_od)

    for segment, trip_key in segments.items():
        segment_od = apply_segment_coeffs(od_df, spec, locals_dict, segment)
        trips = zones[trip_key]

        num_trips = logit(segment_od, trips, trace=trace_rows is not None)

        trips_df[segment] = num_trips

        if trace_rows is not None:
            logger.debug('writing segment %s trace' % segment)
            trace_rows.index = od_df.index
            tracing.write_csv(segment_od[trace_rows].reset_index(),
                              file_name='segment_od_%s' % segment,
                              transpose=False)

    pipeline.replace_table('trips', trips_df)


def apply_segment_coeffs(od_df, spec, locals_dict, segment):
    """Multiply each target variable in the origin-destination
    DataFrame by a segment-specific coefficient.

    Parameters
    ----------
    od_df : pandas DataFrame
        the raw origin-destination DataFrame
    spec : pandas DataFrame, assignment expressions
    locals_dict : dict
        dictionary of constants, etc.
    segment: str, segment name

    Returns
    -------
    pandas DataFrame
    """
    logger.debug('applying segment coefficients to %s' % segment)

    coeffs = pd.Series(spec[segment].values, index=spec.target)
    evaluated_segment = assign.evaluate_constants(coeffs, locals_dict)
    segment_od = od_df.multiply(evaluated_segment, axis=1)

    return segment_od


def logit(df, trips_series, chooser_col='orig', trace=False):
    """Calculate number of trips taken between each origin-destination pair
    using a logit utility model

    Parameters
    ----------
    df : pandas DataFrame, origin-destination pairs
    trips_series : pandas Series
        number of trips originating from each zone
    chooser_col : str
        name of origin zone column in df
    trace : bool
        whether to save utility and probability calculations back
        to original dataframe

    Returns
    -------
    pandas Series
    """
    assert chooser_col in df.index.names
    trips_series.index.name = chooser_col
    util = np.exp(df.sum(axis=1))  # sum the df rows
    sum_util = util.groupby([chooser_col]).sum()  # sum the utilites for each origin zone
    probs = util / sum_util  # calculate the trip probability for each od pair

    if trace:
        df['utils'] = util
        df['sum_utils'] = np.repeat(np.asanyarray(sum_util), len(sum_util.index))
        df['probs'] = probs

    # Since the trips_series (length == number of zones) index aligns
    # with the probs (length == number of zones ^2) index,
    # pandas can correctly align the rows.
    return probs * trips_series
