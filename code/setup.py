from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('cy_hns_map', ['cy_hns_map.pyx'], include_dirs = [np.get_include()],library_dirs=["/home/rjadams/lib"]),
    ]

setup(
    ext_modules = cythonize(extensions)
    )