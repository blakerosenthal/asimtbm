# This is a pared down version of the ODSkims code found in https://github.com/RSGInc/bca4abm
import logging
from os import path
import openmatrix as omx

from activitysim.core import skim as askim

logger = logging.getLogger(__name__)

SKIMS_KEY = 'aggregate_od_matrices'


def read_skims(zone_index, data_dir, model_settings):
    """Reads OpenMatrix skims

    Parameters
    ----------
    zone_index : pandas Index obj
    data_dir : str
        data directory path
    model_settings : dict

    Returns
    -------
    dictionary of Skims objects
    """

    aggregate_od_matrices = model_settings.get(SKIMS_KEY, None)
    if not aggregate_od_matrices:
        raise RuntimeError("No list %s found in model_settings", SKIMS_KEY)

    skims_dict = {}
    for local_name, omx_file_name in aggregate_od_matrices.items():
        omx_file_path = path.join(data_dir, omx_file_name)

        skims = Skims(name=local_name,
                      omx_file_path=omx_file_path,
                      zone_index=zone_index)

        skims_dict[local_name] = skims

    return skims_dict


def close_skims(locals_dict):
    for local_name, val in locals_dict.items():
        if isinstance(val, Skims):
            logger.info("closing %s" % local_name)
            val.close()


class Skims(object):

    def __init__(self, name, omx_file_path, zone_index):

        self.name = name
        self.skims_dict = {}

        self.omx = omx.open_file(omx_file_path, 'r')
        self.omx_shape = tuple([int(s) for s in self.omx.shape()])
        self.matrices = self.omx.listMatrices()

        logger.info("omx file %s skim shape: %s number of skims: %s" %
                    (name, self.omx_shape, len(self.matrices)))

        self.offset_mapper = askim.OffsetMapper()
        self.set_offset_list(zone_index)

        self.omx_indices = self.offset_mapper.map(zone_index)

        logger.debug("Using skim omx indices %s" % list(self.omx_indices))

    def set_offset_list(self, zone_index):
        """determine mapping of zone ids to skim indices

        If a mapping is provided, use it. Otherwise check if the skim size
        is equal to the number of zones, and use basic 1-based indices if not.
        """
        mappings = self.omx.list_mappings()
        if mappings:
            assert len(mappings) == 1, \
                "expected one mapping in %s, found %s" % (self.name, mappings)
            logger.info("Using mapping '%s' from %s" % (mappings[0], self.name))
            self.offset_mapper.set_offset_list(list(self.omx.mapping(mappings[0])))
            return

        if self.omx_shape[0] == len(zone_index):
            logger.info("Using zones list for %s map" % self.name)
            self.offset_mapper.set_offset_list(list(zone_index))
        else:
            logger.info("Using 1-based zone indexing for %s" % self.name)
            self.offset_mapper.set_offset_int(1)

    def __getitem__(self, key):
        """accessor to return pandas Series with specified key

        this allows the skim series (with a useable index) to be
        accessed from expressions as skim['DISTANCE']

        also caches series as a column in self.skims_dict
        """

        assert key in self.matrices

        if key in self.skims_dict:
            omx_data = self.skims_dict[key]
        else:
            omx_data = self.read_from_omx(key)
            self.skims_dict[key] = omx_data

        return omx_data

    def read_from_omx(self, key):
        """selects only the rows and columns that match the od_index to
        avoid unnecessarily reading potentially large matrices into memory

        Returns
        -------
        flattened 2D array
        """
        try:
            data = self.omx[key][self.omx_indices, :][:, self.omx_indices]
        except omx.tables.exceptions.NoSuchNodeError:
            raise RuntimeError("Could not find skim with key '%s' in %s" % (key, self.name))

        return data.flatten()

    def close(self):

        unused_skims = list(set(self.matrices)-set(self.skims_dict.keys()))
        if unused_skims:
            logger.debug("Did not ever use skims %s in '%s'" % (unused_skims, self.name))

        self.omx.close()
        self.skims_dict = {}
