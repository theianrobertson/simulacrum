#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from simulacrum.dataset import create, validate_type_dict, default_coltypes, TYPE_FUNCTIONS

def test_validate_type_dict():
    for value in ('num','int','norm','exp','bin','pois','txt','name','addr',
        'date','uuid','faker'):
        good_dict = {'type': value}
        validate_type_dict(good_dict)
    for value in ('hey', '', 'dict', 'bad_value', 1, None):
        bad_dict = {'type': value}
        with pytest.raises(ValueError):
            validate_type_dict(bad_dict)
    with pytest.raises(ValueError):
        validate_type_dict({'a': 1})

def test_default_coltypes():
    coltypes = default_coltypes()
    for type_dict in coltypes.values():
        validate_type_dict(type_dict)
    assert len(coltypes) == len(TYPE_FUNCTIONS) - 1

def test_blank_create():
    test_df = create()
    assert len(test_df) == 100
    assert len(test_df.columns) == len(TYPE_FUNCTIONS) - 1

def test_cols_types_create():
    test_df = create(length=10, cols=['int'], types=[{'type': 'int'}])
    assert len(test_df) == 10
    assert list(test_df.columns) == ['int']

def test_coltypes_create():
    test_df = create(length=10, coltypes={'int': {'type': 'int'}})
    assert len(test_df) == 10
    assert list(test_df.columns) == ['int']

def test_create_passthrough_params():
    test_df = create(
        length=1000,
        coltypes={
            'int': {'type': 'int', 'min': 10},
            'txt': {'type': 'txt', 'max_nb_chars': 20}
            })
    assert len(test_df) == 1000
    assert set(list(test_df.columns)) == set(['int', 'txt'])
    assert test_df['int'].min() >= 10
    assert test_df['int'].max() <= 100
    lengths = test_df['txt'].apply(lambda x: len(x))
    assert lengths.max() <= 20


def test_create_passthrough_bad_params():
    with pytest.raises(TypeError):
        test_df = create(
            length=1000,
            coltypes={
                'int': {'type': 'int', 'min': 10, 'bad_param': -999},
                'txt': {'type': 'txt', 'max_nb_chars': 20}
                })
