"""Functions for creating a dataframe based on setup dictionaries"""

import logging
import pandas as pd

from simulacrum import types as sim_types

TYPE_FUNCTIONS = {
    'num': sim_types.num_data,
    'int': sim_types.num_int,
    'norm': sim_types.norm_data,
    'exp': sim_types.exp_data,
    'bin': sim_types.binom_data,
    'pois': sim_types.poisson_data,
    'txt': sim_types.text_data,
    'name': sim_types.name_data,
    'addr': sim_types.address_data,
    'date': sim_types.date_data,
    'coords': sim_types.coords_data,
    'uuid': sim_types.uuid_data,
    'categorical': sim_types.categorical_data,
    'faker': sim_types.faker_data}

def help_type(function_name=None):
    to_print = TYPE_FUNCTIONS.items()
    if function_name:
        to_print = [(function_name, TYPE_FUNCTIONS[function_name])]
    for function_name, function in to_print:
        print(
            'Name: {}, Function: simulacrum.types.{}'.format(function_name, function.__name__))
        print(function.__doc__ + '\n')

def create(length=100, cols=None, types=None, coltypes=None, null_rate=0):
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
    null_rate : float, optional default 0
        An optional null rate between 0 and 1 to apply to the entire dataframe.  You can also pass
        a null_rate as a parameter on any type dictionary to override (or only set that column to
        null).

    Returns
    -------
    pandas.DataFrame
        The generated dataframe.
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
        series_null_rate = type_dict.pop('null_rate', null_rate)
        data_builder = TYPE_FUNCTIONS[type_dict.pop('type')]
        series_res[col] = sim_types.null_mask(length, data_builder, series_null_rate, **type_dict)

    return pd.DataFrame(series_res)


def validate_type_dict(type_dict):
    """Validates a type dictionary

    Parameters
    ----------
    type_dict : dict
        Dictionary defining the type to be used for mocking data.

    Raises
    ------
    ValueError
        If the type_dict doesn't pass validations
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
