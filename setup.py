from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_module = Extension(
        "optimized_func",
        ["optimized_func.pyx"],
        extra_compile_args = ["-O3", "-ffast-math", "-march=native",'-fopenmp', ],
        extra_link_args = ['-fopenmp'], 
        )

setup(
    name = "optimized_func",
    cmdclass = { 'build_ext' : build_ext },
    ext_modules = [ext_module],
)
