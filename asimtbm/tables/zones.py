import logging
import os
import pandas as pd

from activitysim.core import inject, config, tracing

logger = logging.getLogger(__name__)

ZONE_LABEL = 'zone'
TABLES_YAML = 'tables.yaml'
TABLE_FILENAMES_KEY = 'aggregate_zone_file_names'


@inject.table()
def zones():
    """ActivitySim pipeline table of raw zone data.

    Reads zone filenames from tables.yaml and combines
    into single table which is then registered to the pipeline.
    Each zone file must be the same length and given a 'zone'
    index label. If no 'zone' column is found, row numbers will
    be used for the zone index.
    """
    table_settings = config.read_model_settings(TABLES_YAML)
    zone_tables = read_zone_tables(table_settings)
    zones_df = combine_zone_tables(zone_tables)

    inject.add_table('zones', zones_df)

    return zones_df


def read_zone_tables(table_settings):
    logger.info('reading tables from configs...')

    table_filenames = table_settings.get(TABLE_FILENAMES_KEY)
    tables = [read_zone_indexed_csv_file(f) for f in table_filenames]

    logger.info('finished reading tables.')

    return tables


def combine_zone_tables(zone_tables):
    logger.info('building aggregate zones table ...')

    # verify that every zone file contains the same zones
    comparison_index = zone_tables[0].index
    for table in zone_tables:
        if not table.index.equals(comparison_index):
            raise RuntimeError(
                "table with columns %s does not match other zones" % table.columns.values)

    combined_zones_df = pd.concat(zone_tables, axis=1)
    logger.info('finished building aggregate zones table with %d zones and columns: %s',
                len(combined_zones_df.index),
                combined_zones_df.columns.values)

    return combined_zones_df


def read_zone_indexed_csv_file(file_name):
    logger.info('reading file \'%s\'' % file_name)

    fpath = config.data_file_path(file_name, mandatory=True)
    zone_df = pd.read_csv(fpath, header=0, comment='#')

    if ZONE_LABEL in zone_df.columns:
        zone_index = ZONE_LABEL  # str
    else:
        # use row numbers for zone ids. convert to 1-based zone ids simply by adding 1
        zone_index = zone_df.index + 1  # Series

    zone_df.set_index(zone_index, drop=True, inplace=True)
    zone_df.index.name = ZONE_LABEL

    return zone_df
