import logging

logger = logging.getLogger(__name__)


def trace_filter(df, trace_od, orig='orig', dest='dest'):
    """Filter out rows from DataFrame matching the trace_od

    Chooses rows matching trace_od. Specify just 'o' to trace
    all trips originating in that zone, just 'd' to trace all
    trips ending on that zone, or both to trace the single trip
    starting in 'o' and ending in 'd'.

    Parameters
    ----------
    df : pandas DataFrame with columns matching orig, dest
    trace_od : list or dict
        list of length == 2 or dict with keys 'o', 'd'

    Returns
    -------
    pandas Series or None
        non-indexed pandas filter. None if no trace_od or parsing error.
    """
    if not trace_od:
        return None

    if isinstance(trace_od, list) and len(trace_od) == 2:
        o, d = trace_od
    elif isinstance(trace_od, dict):
        o, d = trace_od.get('o'), trace_od.get('d')
    else:
        logger.warn("trace_od must be either a list or dict with keys 'o' and 'd'")

    if o and d:
        return (df.loc[:, orig] == o) & (df.loc[:, dest] == d)

    if o:
        return df.loc[:, orig] == o

    if d:
        return df.loc[:, dest] == d

    logger.warn("failed to parse trace_od %s" % trace_od)
    return None
