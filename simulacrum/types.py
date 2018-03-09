"""Functions for generating random data."""

from uuid import uuid4
import logging
import datetime
import numpy as np
import pandas as pd
from faker import Faker

FAKE = Faker()
DT_CLASSES = (datetime.datetime, datetime.date)

def name_data(length):
    """Faker names series"""
    return pd.Series([FAKE.name() for _ in range(length)])

def text_data(length, max_nb_chars=200):
    """Faker text series"""
    return pd.Series([FAKE.text(max_nb_chars) for _ in range(length)])

def address_data(length):
    """Faker address series"""
    return pd.Series([FAKE.address() for _ in range(length)])

def num_data(length, min=0, max=1):
    """Uniform distribution"""
    return pd.Series(np.random.uniform(min, max, length))

def num_int(length, min=0, max=100):
    """Random integers"""
    return pd.Series(np.random.random_integers(min, max, length))

def norm_data(length, mean=0, sd=1):
    """Normal distribution data"""
    return pd.Series(np.random.normal(mean, sd, length))

def exp_data(length, lam=1.0):
    """Exponential distribution data"""
    scale = 1.0 / lam
    return pd.Series(np.random.exponential(scale, length))

def binom_data(length, n=100, p=0.1):
    """Binomial distribution data"""
    return pd.Series(np.random.binomial(n, p, length))

def poisson_data(length, lam=1.0):
    return pd.Series(np.random.poisson(lam, length))

def date_data(length, begin=None, end=None, tzinfo=None):
    """Dates between a start and end.  If no start and end are provided,
    default to one year ago and today.

    Parameters
    ----------
    begin : datetime.datetime, datetime.date or str, optional
        Beginning datetime - will attempt to parse as yyyy-mm-dd if string.  If
        not provided will default to today's date minus 365 days.
    end : datetime.datetime, datetime.date or str, optional
        Ending datetime - will attempt to parse as yyyy-mm-dd is string.  If
        not provided will default to today's date.
    tzinfo : timezone, instance of datetime.tzinfo subclass
        Optional timezone
    """
    if begin is None and end is None:
        datetime_end = datetime.datetime.now()
        datetime_start = datetime_end - datetime.timedelta(365)
    elif isinstance(begin, DT_CLASSES) and isinstance(end, DT_CLASSES):
        datetime_end = end
        datetime_start = begin
    else:
        try:
            datetime_start = datetime.datetime.strptime(begin, '%Y-%m-%d')
            datetime_end = datetime.datetime.strptime(end, '%Y-%m-%d')
        except:
            logging.error('Bad date format, expected yyyy-mm-dd!')
            raise ValueError('Could not parse dates')
    return pd.Series(
        [FAKE.date_time_between_dates(
            datetime_start, datetime_end) for _ in range(length)])

def coords_data(length, lat_min=-90, lat_max=90, lon_min=-180, lon_max=180):
    """Randomly-selected geographic coordinates
    
    Parameters
    ----------
    length : int
        Length of the series
    lat_min : numeric, optional
        Minimum latitude
    lat_max : numeric, optional
        Maximum latitude
    lon_min : numeric, optional
        Minimum longitude
    lon_max : numeric, optional
        Maximum longitude
    """
    if lat_min < -90 or lat_max > 90 or lat_min > lat_max:
        raise ValueError(
            'lat ranges unacceptable; not in [-90, 90] or lat_min > lat_max')
    if lon_min < -180 or lon_max > 180 or lon_min > lon_max:
        raise ValueError(
            'lon ranges unacceptable; not in [-180, 180] or lon_min > lon_max')
    return pd.Series(list(zip(np.random.uniform(lat_min, lat_max, length),
                         np.random.uniform(lat_min, lat_max, length))))

def uuid_data(length):
    """Generate a column of random uuids.

    Parameters
    ----------
    length : int
        Length of the series
    
    Returns
    -------
    pandas.Series
        The column of uuids.
    """
    return pd.Series(list(map(lambda _: uuid4(), range(length))))


def faker_data(length, **kwargs):
    """Generate a column based on any faker data type.

    Parameters
    ----------
    length : int
        Length of the series to return
    kwargs : dict
        A configuration for the faker data. Must contain at least provider () and related args as
        dict.

    Returns
    -------
    pandas.Series
    """
    try:
        provider = kwargs["provider"]
        del kwargs["provider"]
        func = getattr(FAKE, provider)
    except KeyError:
        raise KeyError("You have to define the Faker provider.")
    except AttributeError:
        raise AttributeError("Faker().{}() is not a valid Faker provider.".format(provider))
    return pd.Series(map(lambda _: func(**kwargs), range(length)))


def null_mask(length, type_function, null_rate=0, **kwargs):
    """Masks out a random subset of series values with Numpy nulls (np.nan).  The number of null
    values will be int(null_rate * length)
    
    Parameters
    ----------
    type_function : function
        A function which returns a Pandas series
    null_rate : float, optional
        Optional null rate between 0 and 1 inclusive.
    """
    if not 0 <= null_rate <= 1:
        raise ValueError('null_rate must be between 0 and 1')
    results = type_function(length, **kwargs)
    sample_index = results.sample(int(null_rate * length)).index
    results.loc[results.index.isin(sample_index)] = np.nan
    return results
