from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='ceffyl',
    version='1.2',
    description=('Software to rapidly and flexibly analyse Pulsar Timing ' +
                 'Array data via the Generalised Factorised Likelihood (GFL)' +
                 'method'),
    author='William G. Lamb',
    author_email='william.g.lamb@vanderbilt.edu',
    packages=['ceffyl', 'ceffyl.bw'],
    ext_modules=cythonize("ceffyl/bw/cbandwidths.pyx"),
    include_dirs=[numpy.get_include()],
    package_data={'cbandwidths': ['*']}
)
