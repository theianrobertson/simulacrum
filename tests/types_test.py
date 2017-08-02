import datetime
import pytest
from uuid import UUID
from functools import reduce
import numpy as np

from simulacrum import types
from simulacrum import dataset

def _default_test(listToCheck, typeToWait, length):
    return len(listToCheck) == length\
        and reduce(lambda p, n: p and type(n) == typeToWait, listToCheck, True)

def test_uuid_data():
    """Test uuid data."""
    uuids_list = types.uuid_data(30)
    assert _default_test(uuids_list, UUID, 30)\
        and len(set(uuids_list)) == len(uuids_list)

def test_faker_data_ipv6():
    """Test faker data."""
    ipv6_list = types.faker_data(**{
        "provider": "ipv6",
        "network": False
    }, length=23)
    random_element_list = types.faker_data(**{
        "provider": "random_element",
        "elements": ('a', 'b', 'c', 'd'),
    }, length=13)
    assert _default_test(ipv6_list, str, 23)\
        and _default_test(random_element_list, str, 13)\
        and reduce(lambda p, n: p and n in ['a', 'b', 'c', 'd'], random_element_list, True)

def test_name_data():
    names = types.name_data(10)
    assert len(names) == 10
    assert names.dtype == np.dtype('O')
    for item in names:
        assert isinstance(item, str)

def test_text_data():
    text = types.text_data(10)
    assert len(text) == 10
    assert text.dtype == np.dtype('O')
    for item in text:
        assert isinstance(item, str)
    text_max_10 = types.text_data(10, 10)
    assert len(text_max_10) == 10
    lengths = text_max_10.apply(lambda x: len(x))
    assert lengths.max() <= 10

def test_address_data():
    addresses = types.address_data(10)
    assert len(addresses) == 10
    assert addresses.dtype == np.dtype('O')
    for item in addresses:
        assert isinstance(item, str)

def test_num_data():
    nums = types.num_data(1000, minimum=0, maximum=1)
    assert len(nums) == 1000
    assert nums.max() <= 1
    assert nums.min() >= 0
    assert nums.dtype == np.dtype('float')

def test_num_int():
    nums = types.num_int(1000, minimum=0, maximum=100)
    assert len(nums) == 1000
    assert nums.max() <= 100
    assert nums.min() >= 0
    assert nums.dtype == np.dtype('int')

def test_norm_data():
    nums = types.norm_data(1000, mean=0, stdev=100)
    assert len(nums) == 1000
    assert nums.dtype == np.dtype('float')

def test_exp_data():
    nums = types.exp_data(1000, scale=10)
    assert len(nums) == 1000
    assert nums.dtype == np.dtype('float')
    assert nums.max() >= 0

def test_binom_data():
    nums = types.binom_data(10)
    assert len(nums) == 10
    assert nums.dtype == np.dtype('int')

def test_poisson_data():
    nums = types.poisson_data(10)
    assert len(nums) == 10
    assert nums.dtype == np.dtype('int')

def test_date_data():
    dates = types.date_data(length=1000)
    assert len(dates) == 1000
    assert len(dates.unique()) > 1
    assert dates.dtype == np.dtype('<M8[ns]')
    
def test_date_data_between():
    dates = types.date_data(
        length=1000,
        datetime_start=datetime.datetime(2000, 1, 1),
        datetime_end=datetime.datetime(2000, 12, 31))
    assert len(dates) == 1000
    assert len(dates.unique()) > 1
    assert dates.min() >= datetime.datetime(2000, 1, 1)
    assert dates.max() < datetime.datetime(2001, 1, 1)
    assert dates.dtype == np.dtype('<M8[ns]')

def test_coords_data():
    coords = types.coords_data(10)
    assert len(coords) == 10
    assert coords.dtype == np.dtype('O')
    for item in coords:
        assert isinstance(item, tuple)
    with pytest.raises(ValueError):
        coords = types.coords_data(10, lat_min=-91)
    with pytest.raises(ValueError):
        coords = types.coords_data(10, lat_max=91)
    with pytest.raises(ValueError):
        coords = types.coords_data(10, lon_min=-181)
    with pytest.raises(ValueError):
        coords = types.coords_data(10, lon_max=181)
