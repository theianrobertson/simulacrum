"""Functions for generating random data."""

import logging
import pandas as pd

from simulacrum import types

TYPE_FUNCTIONS = {
    'num': types.num_data,
    'int': types.num_int,
    'norm': types.norm_data,
    'exp': types.exp_data,
    'bin': types.binom_data,
    'pois': types.poisson_data,
    'txt': types.text_data,
    'name': types.name_data,
    'addr': types.address_data,
    'date': types.date_data,
    'coords': types.coords_data,
    'uuid': types.uuid_data,
    'faker': types.faker_data}


def create(length=100, cols=None, types=None, coltypes=None):
    """Create a dataset based on passed in information.

    A user must either pass in cols and types lists, OR coltypes, OR the
    function will default to "one of each" type.

    Parameters
    ----------
    length : int, optional
        How many records (rows) in the returned dataframe
    cols : list, optional
        A list of column names
    types : list of dict, optional
        A list of "type dictionaries", defining both the type, and any
        parameters to be passed to the type function.
    coltypes : dict, optional
        A combined version of cols and types, where the keys of the dictionary
        are cols, and the values are type dictionaries.
    """
    if cols and types and coltypes:
        raise ValueError(
            'coltypes should not be defined when cols and types are defined')
    elif cols is not None and types is not None:
        column_iter = cols
        type_iter = types
        if len(column_iter) != len(type_iter):
            raise ValueError(
                'cols and types must be lists of equal length')
    else:
        if coltypes is None:
            logging.warning('No coltypes specified, using default')
            coltypes = default_coltypes()
        column_iter = coltypes.keys()
        type_iter = coltypes.values()

    series_res = {}
    for col, type_dict in zip(column_iter, type_iter):
        validate_type_dict(type_dict)
        data_builder = TYPE_FUNCTIONS[type_dict['type']]
        del type_dict['type']
        series_res[col] = data_builder(length, **type_dict)

    return pd.DataFrame(series_res)


def validate_type_dict(type_dict):
    """Validates a type dictionary

    Parameters
    ----------
    type_dict : dict
        Dictionary defining the type to be used for mocking data.
    """
    if "type" not in type_dict:
        logging.error('Missing "type": %s', str(type_dict))
        raise ValueError('"type" is a required key for all type dicts')
    if type_dict['type'] not in TYPE_FUNCTIONS:
        logging.error('Bad "type": %s', str(type_dict))
        raise ValueError('"{}" is not a valid type'.format(type_dict['type']))

def default_coltypes():
    """Sets up the default column types
    
    Returns
    -------
    dict
        A dictionary that can be passed in to coltypes in create function,
        which just creates one column of each type in TYPE_FUNCTIONS
    """
    coltypes = {}
    for key in TYPE_FUNCTIONS:
        if key != 'faker':
            coltypes[key] = {'type': key}
    return coltypes
