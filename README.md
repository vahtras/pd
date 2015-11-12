### Installation

Export pythonpath to point to the root directory of pd.

Assuming it lies in the home directory:

```sh
$ export PYTHONPATH=/home/$USER/pd":$PYTHONPATH
```


### Cython dependency

To use cython for speedup, run the following command in the root directory:


```sh
$ python setup.py build_ext --inplace
```

This will leave a .so object file which is imported under the hood.


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
