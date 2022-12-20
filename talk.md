# Data handling: pandas

## Olav Vahtras

Computational Python

---

layout: false

## Datahandling in Python

### The `pandas` module

Setup:

```
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import seaborn

```

Two main data structures

* Series
* Data frames
---

### Series

One-dimensional labeled data

```
>>> s = pd.Series([0.1, 0.2, 0.3, 0.4])
>>> print(s)
0    0.1
1    0.2
2    0.3
3    0.4
dtype: float64

```
--
```
>>> print(s.index)
RangeIndex(start=0, stop=4, step=1)


```
--
```
>>> print(s.values)
[0.1 0.2 0.3 0.4]

```

---

* indices can be labels (like a dict with order)

```
>>> s = pd.Series(np.arange(4), index=['a', 'b', 'c', 'd'])
>>> print(s)
a    0
b    1
c    2
d    3
dtype: int64
>>> print(s['d'])
3
>>>
```
--
* Initialize with dict

```
>>> s = pd.Series({'a': 1, 'b': 2, 'c': 3, 'd': 4})
>>> print(s)
a    1
b    2
c    3
d    4
dtype: int64
>>>
```
--
* Indexing as a dict

```
>>> print(s['a'])
1

```
---

* Elementwise operations
```
>>> print(s * 100)
a    100
b    200
c    300
d    400
dtype: int64
>>>
```
--

* Slicing
```
>>> s['b': 'c']
b    2
c    3
dtype: int64
>>>
```

---

* List indexing
```
>>> print(s[['b', 'c']])
b    2
c    3
dtype: int64
>>>
```
--

* Bool indexing
```
>>> print(s[s>2])
c    3
d    4
dtype: int64
>>>
```
--

* Other operations
```
>>> s.mean()
2.5
>>>
```
---

* Alignment on indices
```
>>> s['a':'b'] + s['b':'c']
a    NaN
b    4.0
c    NaN
dtype: float64
>>>
```
---

### DataFrames

* Tabular data structure (like spreadsheet, sql table)
* Multiple series with common index

```
>>> data = {'country': ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom'],
...        'population': [11.3, 64.3, 81.3, 16.9, 64.9],
...        'area': [30510, 671308, 357050, 41526, 244820],
...        'capital': ['Brussels', 'Paris', 'Berlin', 'Amsterdam', 'London']}
>>>
```
--
```
>>> countries = pd.DataFrame(data)
>>> print(countries)
          country  population    area    capital
0         Belgium        11.3   30510   Brussels
1          France        64.3  671308      Paris
2         Germany        81.3  357050     Berlin
3     Netherlands        16.9   41526  Amsterdam
4  United Kingdom        64.9  244820     London
>>>
```

---

* Attributes: index, columns, dtypes, values

```
>>> countries.index
RangeIndex(start=0, stop=5, step=1)
>>>
```
--
```
>>> countries.columns
Index(['country', 'population', 'area', 'capital'], dtype='object')
>>>
```
--
```
>>> countries.dtypes
country        object
population    float64
area            int64
capital        object
dtype: object
>>>
```
--
```
>>> countries.values
array([['Belgium', 11.3, 30510, 'Brussels'],
       ['France', 64.3, 671308, 'Paris'],
       ['Germany', 81.3, 357050, 'Berlin'],
       ['Netherlands', 16.9, 41526, 'Amsterdam'],
       ['United Kingdom', 64.9, 244820, 'London']], dtype=object)

```
---

* Info

```
>>> countries.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 4 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   country     5 non-null      object 
 1   population  5 non-null      float64
 2   area        5 non-null      int64  
 3   capital     5 non-null      object 
dtypes: float64(1), int64(1), object(2)
memory usage: 288.0+ bytes

```

---

* Set a column as index

```
>>> print(countries)
          country  population    area    capital
0         Belgium        11.3   30510   Brussels
1          France        64.3  671308      Paris
2         Germany        81.3  357050     Berlin
3     Netherlands        16.9   41526  Amsterdam
4  United Kingdom        64.9  244820     London

```
--

```
>>> countries = countries.set_index('country')
>>>
```
--

```
>>> print(countries)
                population    area    capital
country                                      
Belgium               11.3   30510   Brussels
France                64.3  671308      Paris
Germany               81.3  357050     Berlin
Netherlands           16.9   41526  Amsterdam
United Kingdom        64.9  244820     London

```

---

* Access a single series in a table
```
>>> print(countries['area'])
country
Belgium            30510
France            671308
Germany           357050
Netherlands        41526
United Kingdom    244820
Name: area, dtype: int64
>>>
```
--
```
>>> print(countries['capital']['France'])
Paris
>>>
```
--

* Arithmetic (population density)
```
>>> print(countries['population']/countries['area']*10**6)
country
Belgium           370.370370
France             95.783158
Germany           227.699202
Netherlands       406.973944
United Kingdom    265.092721
dtype: float64
>>>
```

---


* Add new column

```
>>> countries['density'] =  countries['population']/countries['area']*10**6
>>> print(countries)
                population    area    capital     density
country                                                  
Belgium               11.3   30510   Brussels  370.370370
France                64.3  671308      Paris   95.783158
Germany               81.3  357050     Berlin  227.699202
Netherlands           16.9   41526  Amsterdam  406.973944
United Kingdom        64.9  244820     London  265.092721

```

--

* Filter data

```
>>> print(countries[countries['density'] > 300])
             population   area    capital     density
country                                              
Belgium            11.3  30510   Brussels  370.370370
Netherlands        16.9  41526  Amsterdam  406.973944

```
---

* Sort data

```
>>> print(countries.sort_values('density', ascending=False))
                population    area    capital     density
country                                                  
Netherlands           16.9   41526  Amsterdam  406.973944
Belgium               11.3   30510   Brussels  370.370370
United Kingdom        64.9  244820     London  265.092721
Germany               81.3  357050     Berlin  227.699202
France                64.3  671308      Paris   95.783158

```

--

* Statistics

```
>>> print(countries.describe())
       population           area     density
count    5.000000       5.000000    5.000000
mean    47.740000  269042.800000  273.183879
std     31.519645  264012.827994  123.440607
min     11.300000   30510.000000   95.783158
25%     16.900000   41526.000000  227.699202
50%     64.300000  244820.000000  265.092721
75%     64.900000  357050.000000  370.370370
max     81.300000  671308.000000  406.973944

```

---

### Features

* like numpy arrays with labels
* supported import/export formats: CSV, SQL, Excel...
* support for missing data
* support for heterogeneous data
* merging data
* reshaping data
* easy plotting 


---

### Example

A movie database in csv format `titles.csv`

* Titles

```bash
$ head titles.csv
title,year
The Rising Son,1990
Ashes of Kukulcan,2016
The Thousand Plane Raid,1969
Crucea de piatra,1993
The 86,2015
Gaiking II,2011
Medusa (IV),2015
Cultural Menace,2004
Alex in Wonderland,1970
```

--

```
>>> titles = pd.read_csv('titles.csv')
>>> titles.head()
                     title  year
0           The Rising Son  1990
1        Ashes of Kukulcan  2016
2  The Thousand Plane Raid  1969
3         Crucea de piatra  1993
4                   The 86  2015
>>>
```
---

* How many movies?

--

```
>>> len(titles)
215981
>>>
```

--

* Two earliest films?

--


```
>>> titles.sort('year').head(2)
                                                    title  year
165182                                         Miss Jerry  1894
85708   Reproduction of the Corbett and Fitzsimmons Fight  1897
>>>
```
--

* Number of movies with title *Hamlet*?

--

```
>>> len(titles[titles.title == "Hamlet"])
19
>>>
```

---

* List all *Treasure Island* movies sorted by year

--

```
>>> titles[titles.title == "Treasure Island"].sort('year')
                  title  year
191379  Treasure Island  1918
47769   Treasure Island  1920
192917  Treasure Island  1934
90175   Treasure Island  1950
104714  Treasure Island  1972
103646  Treasure Island  1973
190792  Treasure Island  1985
166675  Treasure Island  1999
>>>
```

---

#### Acknowledgement 
Based on [EuroSciPy tutorial](https://github.com/jorisvandenbossche/2015-EuroScipy-pandas-tutorial.git) by Joris Van den Bossche
and [PyCon tutorial](https://github.com/brandon-rhodes/pycon-pandas-tutorial) by Brandon Rhodes
