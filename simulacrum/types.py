"""Functions for generating random data."""

from uuid import uuid4
import logging
import sys
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

def num_data(length, minimum=0, maximum=1):
    """Uniform distribution"""
    return pd.Series(np.random.uniform(minimum, maximum, length))

def num_int(length, minimum=0, maximum=100):
    """Random integers"""
    return pd.Series(np.random.random_integers(minimum, maximum, length))

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
    """dates between a start and end.  If no start and end are provided,
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
    """Geographic coordinates"""
    if lat_min < -90 or lat_max > 90 or lat_min > lat_max:
        raise ValueError(
            'lat ranges unacceptable; not in [-90, 90] or lat_min > lat_max')
    if lon_min < -180 or lon_max > 180 or lon_min > lon_max:
        raise ValueError(
            'lon ranges unacceptable; not in [-180, 180] or lon_min > lon_max')
    return pd.Series(list(zip(np.random.uniform(lat_min, lat_max, length),
                         np.random.uniform(lat_min, lat_max, length))))

def uuid_data(length):
    """
    Generate a column of random uuids.

    :param length: The number of uuids.
    :type length: int.
    :return: The column of uuids.
    :rtype: pd.Series

    """
    return pd.Series(list(map(lambda _: uuid4(), range(length))))


def faker_data(length, **kwargs):
    """
    Generate a column based on any faker data type.

    :param kwargs: A configuration for the faker data. Must contain faker provider and related args as dict.
    :param length: The number of rows wanted.
    :type length: int.
    :return: The column of Faker data.
    :rtype: pd.Series

    """
    try:
        provider = kwargs["provider"]
        del kwargs["provider"]
        return pd.Series(list(map(
            lambda _: getattr(FAKE, provider)(**kwargs), range(length))))
    except KeyError:
        raise KeyError("You have to define the Faker provider.")
    except AttributeError:
        raise AttributeError("Faker().{}() is not a valid Faker provider.".format(provider))
