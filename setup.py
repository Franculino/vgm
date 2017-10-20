from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["core/g_math.pyx", "core/g_output.pyx", "core/physiology.pyx", "core/run_faster.pyx"]),
    include_dirs = [numpy.get_include()]
)
