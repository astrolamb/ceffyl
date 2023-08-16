# ceffyl
[![PyPI version](https://badge.fury.io/py/ceffyl.svg)](https://badge.fury.io/py/ceffyl)
[![conda-forge](https://anaconda.org/conda-forge/ceffyl/badges/version.svg)](https://anaconda.org/conda-forge/ceffyl/)
[![DOI](https://zenodo.org/badge/474781623.svg)](https://zenodo.org/badge/latestdoi/474781623)

Pronounced /ˈkɛfɨ̞l/ **('keff-ill')**, meaning 'horse' in Cymraeg/Welsh 🏴󠁧󠁢󠁷󠁬󠁳󠁿🐎 

A software package to rapidly and flexibly analyse pulsar timing array (PTA) data via refiting to pulsar timing free spectra.

This can be done by fitting to a free spectrum of the entire PTA or to individual pulsars!

## Installation

To install via `pip', simply use PyPi:
```bash
pip install ceffyl
```

To install via Anaconda:
```bash
conda install -c conda-forge ceffyl
```

## data
Download representations of PTA data to accurately fit spectral models with ceffyl!

- 🆕 **NANOGrav 15-year data set** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8060824.svg)](https://doi.org/10.5281/zenodo.8060824)
    - [PTA free spectrum refit data](https://zenodo.org/record/8060824)

- **IPTA DR2** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5787557.svg)](https://doi.org/10.5281/zenodo.5787557)
    - [PTA free spectrum refit data](https://github.com/astrolamb/ceffyl/tree/main/data/IPTA_DR2)

- **NANOGrav 12.5-year data set** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4312297.svg)](https://doi.org/10.5281/zenodo.4312297)
    - [PTA free spectrum refit data (zipped)](https://nanograv.org/science/data/nanograv-125y-kde-representation-ceffyl)

## examples

- [**PTA free spectrum refit example**](https://github.com/astrolamb/ceffyl/blob/main/examples/PTA_freespec_ex1.ipynb)
    - This is the **fastest** and **most accurate** refit technique. Fit any GWB spectrum that you'd like in < 5 minutes!
    
- [**GFL Lite refit example**](https://github.com/astrolamb/ceffyl/blob/main/examples/gfl_lite_ex2.ipynb)
    - Fit GWB models quickly and accurately to different combinations of pulsars!
    
- [**GFL refit example**](https://github.com/astrolamb/ceffyl/blob/main/examples/gfl_ex3.ipynb)
    - Fit GWB and custom intrinsic red noise models to different pulsars quickly! **Experimental** - use with caution!

Do you have your own free spectrum posteriors that you want to work in ceffyl? Learn about making your own KDE posteriors [here](https://github.com/astrolamb/ceffyl/tree/main/examples)

## Attribution

```bash
@misc{lamb2023need,
      title={The Need For Speed: Rapid Refitting Techniques for Bayesian Spectral Characterization of the Gravitational Wave Background Using PTAs}, 
      author={William G. Lamb and Stephen R. Taylor and Rutger van Haasteren},
      year={2023},
      eprint={2303.15442},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE}
}
```
