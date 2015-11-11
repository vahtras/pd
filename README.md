### Cython dependency

To use cython for speedup, run the following command in the root directory:


```sh
$ python setup.py build_ext --inplace
```

This will leave a .so object file which can be imported by particles.
