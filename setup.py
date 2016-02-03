#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils import build_ext
from setuptools import setup, Extension

import numpy

ext_module = Extension(
        "pd.optimized_func",
        ["pd/optimized_func.pyx"],
        extra_compile_args = ["-O3", "-ffast-math", "-march=native",'-fopenmp', ],
        extra_link_args = ['-fopenmp'], 
        include_dirs = [ numpy.get_include()],
        )

setup(name = "pd",
    version = "0.1",
    packages = [ 'pd',],
    ext_modules = [ext_module],
    cmdclass = { 'build_ext' : build_ext },
    setup_requires = [ 'numpy==1.9.2', 'cython==0.20.1' ],
    install_requires = [ 'numpy==1.9.2', 'cython==0.20.1' ],
)
