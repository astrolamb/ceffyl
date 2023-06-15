from setuptools import setup

from Cython.Build import cythonize
import numpy

setup(
    name='ceffyl',
    version='1.25',
    description=('Software to rapidly and flexibly analyse Pulsar Timing ' +
                 'Array data via the Generalised Factorised Likelihood (GFL)' +
                 'method'),
    author='William G. Lamb',
    author_email='william.g.lamb@vanderbilt.edu',
    packages=['ceffyl', 'ceffyl.bw'],
    ext_modules=cythonize("ceffyl/bw/cbandwidths.pyx"),
    install_requires = [
        "acor@git+https://github.com/davecwright3/acor.git@main",
        "Cython==0.29.33",
        "enterprise_extensions==2.4.2",
        "enterprise_pulsar==3.3.3",
        "h5py==3.7.0",
        "holodeck==0.2.1",
        "kalepy==1.3",
        "KDEpy==1.1.0",
        "la_forge==1.0.2",
        "natsort==8.2.0",
        "numpy==1.23.5",
        "PTMCMCSampler==2.1.1",
        "scipy==1.10.0",
    ],
    include_dirs=[numpy.get_include()],
    package_data={'cbandwidths': ['*']}
)
