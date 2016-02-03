from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


from setuptools import setup, Command

import numpy

ext_module = Extension(
        "optimized_func",
        ["optimized_func.pyx"],
        extra_compile_args = ["-O3", "-ffast-math", "-march=native",'-fopenmp', ],
        extra_link_args = ['-fopenmp'], 
        include_dirs = [ numpy.get_include()],
        )

setup(
    version = "0.1",
    name = "pd",
    ext_modules = [ext_module],
    cmdclass = { 'build_ext' : build_ext },
    setup_requires = [ 'numpy==1.9.2', 'cython==0.20.1' ],
    install_requires = [ 'numpy==1.9.2', 'cython==0.20.1' ],
)
