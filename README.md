# Simulacrum

Simulacrum is a simple way to pass in a dictionary object, with column names and corresponding data types and output a pandas DataFrame of random data. This is great for creating a fake data set or testing a data science script whose validity through generalization needs to be tested. This is still a work in progress. Right now, numerical data can be made to fit a common statistical distribution so long as the proper statistical parameters are included with the type key in the dictionary.

Simulacrum fulfills the following use cases:
- When data is needed for a tutorial, in order to train a model perfectly
- When real data that is well understood cannot be gathered fast enough, but can be simulated quickly
- To develop test case datasets for testing machine learning pipelines and applications on more generalizable data

For instance, companies that operate primarily as brick and mortars cannot gather data at
the same rate as ecommerce companies typically. Gathering customer data may take months. Therefore the data to develop machine learning applications may not exist at
present but the company may have a general understanding of the shape of this data, and can thus simulate it. A model can be
trained on this simulated data and a machine learning application can be developed in parallel to gathering real data! This
hopefully engineers scalability and foresight for companies with slow data velocity.

## Installation

```
$ pip install simulacrum
```

## Usage
```python
import simulacrum as sm
types = {
    'entries': {'type': 'exp', 'lam': 0.5},
    'names': {'type': 'name'},
    'salaries': {'type': 'norm', 'mean': 55000, 'sd': 20000}}

res = sm.create(length=100, coltypes=types)
print(res.head())

#     entries           names       salaries
# 0  0.453003    Terri Fisher   65471.389461
# 1  0.247907    William West   53718.937276
# 2  2.637221    John Johnson   52750.467547
# 3  1.403986       Nancy Kim   50436.549855
# 4  0.437987  Jennifer Moses  116760.941144
```
The `coltypes` parameter is a dict, where the keys are column names, and the
values are dicts with keys `type` and then whatever parameters the type takes. 
Please review source code to see how to properly pass the correct keys and
values. Possible values for the `type` parameter are as follows:

type|Description|Parameters
---|---|---
num|Random floats from a uniform distribution|`min=0`, `max=1`
int|Random integers from a uniform distribution|`min=0`, `max=100`
norm|Random floats from a normal distribution|`mean=0`, `sd=1`
exp|Random floats from an exponential distribution|`lam=1.0`
bin|Random integers from a binomial distribution|`n=100`,`p=0.1`
pois|Random integers from a Poisson distribution|`lam=1.0`
txt|Faker text - sequences of lorem ipsum-style text|`max_nb_chars=200`
name|Faker name - random full names|
addr|Faker address - random full addresses|
date|Dates between begin and end.  If no begin/end are provided, default to the past year.|`begin=None`, `end=None`, `tzinfo=None`
coords|Random tuples of floats from uniform 2D distribution|`lat_min=-90`, `lat_max=90`, `lon_min=-180`, `lon_max=180`
uuid|Randomly selected UUIDs|
faker|Custom faker field - see below|`provider`, `kwargs`

We can also use the ColTypes class to create the coltypes dict with function calls:

```python

import simulacrum as sm

col_types = sm.ColTypes()
col_types.add_coltype('ids', 'uuid')
col_types.add_coltype('name', 'name')
col_types.add_coltype('salaries', 'norm', mean=50000, stdev=1000)
col_types.add_coltype('ips', 'faker', provider='ipv6')

data_set = sm.create(1000, coltypes=col_types.get_coltypes())
```

If you don't specify any types, a dataframe is created with one of each of the available types:

```python
import simulacrum as sm
data_set = sm.create()
list(df.iterrows())[0][1]
#addr      55752 Clifford Crest Apt. 617\nBrownview, CA 7...
#bin                                                      16
#coords                      (-73.4784872076, 38.9711531723)
#date                                    2017-02-15 20:42:52
#exp                                                 2.45768
#int                                                      88
#name                                           Katelyn Tran
#norm                                               -2.36078
#num                                                0.150451
#pois                                                      1
#txt       Possimus voluptas similique vel. Veritatis fac...
#uuid                   64db0551-9a36-4ab7-adc5-942016735f51
#Name: 0, dtype: object
```

### Faker type

If you want to use other data types not allowed in simulacrum by default, you can use the `faker` type to use each data type provided by the awesome faker library: https://faker.readthedocs.io/en/latest/providers.html

To use the faker type, you must pass the provider name and optional args like:

```python
faker_examples = {
    # same as fake.simple_profile(sex='F')
    'names': {'type': 'faker', 'provider': 'simple_profile', 'sex': 'F'},
    # same as fake.ipv6()
    'ips': {'type': 'faker', 'provider': 'ipv6'},
    # same as fake.job()
    'jobs': {'type': 'faker', 'provider': 'job'},
    # same as fake.pydict(nb_elements=10, variable_nb_elements=True)
    'metadata': {'type': 'faker', 'provider': 'pydict', 'nb_elements': 10, 'variable_nb_elements': True},
    'zips': {'type': 'faker', 'provider': 'zipcode'}
}
df = sm.create(coltypes=faker_examples)

# Or with ColTypes
col_types.add_coltype('ids', 'faker', provider='pydict', nb_elements=10, variable_nb_elements=True)
```

### TODO
- Add a function for fake categorical variables
- Handle random seeds some way

### Development

```
$ git clone https://github.com/jbrambleDC/simulacrum.git && cd simulacrum
$ python setup.py install
```
if you want to contribute. Checkout a branch off of master, and open a PR!
