# ceffyl
[![PyPI version](https://badge.fury.io/py/ceffyl.svg)](https://badge.fury.io/py/ceffyl)
[![DOI](https://zenodo.org/badge/474781623.svg)](https://zenodo.org/badge/latestdoi/474781623)

Pronounced /ˈkɛfɨ̞l/ **('keff-ill')**, meaning 'horse' in Cymraeg/Welsh 🏴󠁧󠁢󠁷󠁬󠁳󠁿🐎 

A software package to rapidly and flexibly analyse pulsar timing array (PTA) data via refiting to pulsar timing free spectra.

This can be done by fitting to a free spectrum of the entire PTA or to individual pulsars!

## Installation

To install via `pip', simply use PyPi:
```bash
pip install ceffyl
```

## data
Download representations of PTA data to accurately fit spectral models with ceffyl!

- [NANOGrav 12.5-year data set](https://nanograv.org/science/data/nanograv-125y-kde-representation-ceffyl)

## examples

Do you just want fit your models really quickly? Find examples [in this directory!](https://github.com/astrolamb/ceffyl/tree/main/examples)

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
