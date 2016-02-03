### Pre-requisites:

numpy
cython

### Installation

From source on Linux:

```sh
$> git clone https://github.org/vahtras/pd.git
$> cd pd
$> python setup.py install
```

### Running tests

The recommended way is to execute

```sh
$ nosetests
```

in the pd root directory.

For extensive testing, run 

```sh
$ nosetests -a 'speed=slow'
```
