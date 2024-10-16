""" Setup file for ceffyl. """

from pathlib import Path
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='ceffyl',
    version='1.4',
    description=('Software to rapidly and flexibly analyse Pulsar Timing ' +
                 'Array data via factorised likelihood methods (Lamb et al. 2023)'),
    author='William G. Lamb',
    author_email='william.g.lamb@vanderbilt.edu',
    packages=['ceffyl', 'ceffyl.bw'],
    ext_modules=cythonize("ceffyl/bw/cbandwidths.pyx", include_path=["ceffyl/bw/"]),
    zip_safe=False,
    install_requires=[
        "Cython>=0.29.37,<1.0.0",
        "encor>=1.1.2",
        "enterprise-pulsar>=3.4.1,<4.0.0",
        "enterprise_extensions>=2.4.2,<3.0.0",
        "h5py>=3.11.0,<4.0.0",
        "kalepy>=1.4,<2.0.0",
        "KDEpy>=1.1.0,<2.0.0",
        "la_forge>=1.1.0,<2.0.0",
        "natsort>=8.4.0,<9.0.0",
        "PTMCMCSampler>=2.1.2,<3.0.0",
    ],
    extras_require={"GP": "holodeck-gw>=1.0.0,<2.0.0"},
    include_dirs=[np.get_include()],
    package_data={'cbandwidths': ['*'], "": ["*.pyx"]},
    long_description=long_description,
    long_description_content_type='text/markdown',
)
