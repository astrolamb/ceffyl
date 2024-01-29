from setuptools import setup
from setuptools.command.install import install
from Cython.Build import cythonize
import numpy
import sys
from pathlib import Path
import subprocess
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


"""
class PostInstallCommand(install):
    #Post-installation for installation mode.
    def run(self):
        install.run(self)
        # Install enterprise without dependencies
        subprocess.check_output(['pip', 'install', '--no-deps',
                                 'enterprise-pulsar'])
"""
                                 

setup(
    #cmdclass={'install': PostInstallCommand},
    name='ceffyl',
    version='1.30.4',
    description=('Software to rapidly and flexibly analyse Pulsar Timing ' +
                 'Array data via factorised likelihood methods (Lamb et al. 2023)'),
    author='William G. Lamb',
    author_email='william.g.lamb@vanderbilt.edu',
    packages=['ceffyl', 'ceffyl.bw'],
    ext_modules=cythonize("ceffyl/bw/cbandwidths.pyx", include_path=["ceffyl/bw/"]),
    zip_safe=False,
    install_requires=[
        "encor>=1.1.2",
        "Cython>=0.29.33,<1.0.0",
        "h5py>=3.7.0,<4.0.0",
        "kalepy>=1.3,<2.0.0",
        "KDEpy>=1.1.0,<2.0.0",
        "la_forge>=1.0.2,<2.0.0",
        "natsort>=8.2.0,<9.0.0",
        "numpy>=1.23.5,<2.0.0",
        "PTMCMCSampler>=2.1.1,<3.0.0",
        "scipy>=1.10.0,<2.0.0",
    ],
    extras_require={"GP": "holodeck-gw>=1.0.0,<2.0.0"},
    include_dirs=[numpy.get_include()],
    package_data={'cbandwidths': ['*'], "": ["*.pyx"]},
    long_description=long_description,
    long_description_content_type='text/markdown',
)
