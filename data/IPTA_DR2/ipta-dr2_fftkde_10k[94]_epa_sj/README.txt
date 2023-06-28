ipta-dr2_fftkde_10k[94]_epa_sj data repo
Contains probability densities from a kernel density estimator representing the spatially-uncorrelated PTA free spectrum to be used with the ceffyl software package.
—-------------
How to use: In ceffyl, set the `datadir` path to this directory. Everything else is sorted under the hood
—-------------
Analysis:
Dataset: IPTA DR2 data set - all pulsars

Free spectrum analysis: 30 freq powerlaw intrinsic red noise + 30 freq uncorrelated free spectrum
Based on the free spectrum chain published here: https://zenodo.org/record/5787557

CITE: International Pulsar Timing Array Collaboration. (2022). IPTA DR2 - GWB analysis MCMC output (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5787557

KDE: KDEpy FFTKDE with Epanechnikov kernel, Sheather-Jones bandwidth
—-----------------
density.npy: array of log PDFs extracted from the KDE representations of the PTA free spectrum
log10rholabels.txt: labels for log10rho parameters used in free spectrum analysis
log10rhogrid.npy: grid of log10rho used to extract PDFs from the KDE representations of the free spectrum posteriors
freqs.npy: list of GW frequencies used in analysis
bandwidths.npy: bandwidths of kernels to create the KDEs
pulsar_list.txt: list of pulsars to choose to refit (for PTA free spectrum refit, this is just set to ‘freespec’ as single pulsars cannot be added/removed)


