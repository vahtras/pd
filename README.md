### Cython dependency

To use cython for speedup, run the following command in the root directory:


```sh
$ python setup.py build_ext --inplace
```

This will leave a .so object file which can be imported.
This is taken care of automatically.

### Running tests

The recommended way is to execute

```sh
$nosetests
```

in the pd root directory.

For extensive testing, run 

```sh
$ nosetests -s 'speed=slow'
```
