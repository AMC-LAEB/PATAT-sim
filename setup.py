from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize, build_ext

import numpy as np

with open("README.md", 'r') as f:
    long_description = f.read()

exts = [Extension(name='PATATv1',
                  sources=["PATAT/PATATv1.pyx"],
                  include_dirs=[np.get_include()]),
        Extension(name='PATATv2',
                  sources=["PATAT/PATATv2.pyx"],
                  include_dirs=[np.get_include()])]

setup(
    name="PATAT-sim",
    version="2.0",
    packages=find_packages(),
    author="Alvin X Han, Brooke E Nichols, Colin A Russell",
    author_email = 'x.han@amsterdamumc.nl',
    description="A stochastic agent-based (Propelling Action for Test And Treat) model to simulate SARS-CoV-2 epidemics.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/AMC-LAEB/PATAT-sim",
    ext_modules=cythonize(exts),
    build_ext=build_ext,
    #include_dirs=[np.get_include()],
    setup_requires=["cython>=0.29.23",
                    "numpy>=1.20.2"],
    install_requires=[
        'numpy>=1.20.2',
        'scipy>=1.6.2',
        'pandas>=1.2.4',
        'openpyxl>=3.0.7'
    ],
    scripts = ['bin/runpatat.py']
)
