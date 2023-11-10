# Making KDE density files from free spectra for ceffyl

In this notebook, we'll show a basic tutorial on how to convert free spectrum MCMC posteriors from `enterprise` into Kernel Density Estimators (KDEs) and then into probability density files


```python
%reload_ext autoreload
%autoreload 2
%config InlineBackend.figure_format ='retina'
%matplotlib inline

from ceffyl import densities
```

The easiest way to make highly-optimised, KDE density representations of free spectrum posteriors is to use the `densities.DE_Factory().setup_densities()` method. It has everything that you need built in! We will expand on what this function actually does later, but below you'll find the minimum working code to make a compressed data representation ready to use in `Ceffyl`.

First, load your PTMCMC outputs using `la_forge` and save the posterior as a HDF5-based `.core` file.

For the PTA free spectrum, you'll have one MCMC file.

For the GFL Lite/GFL methods, you'll have $N_p$ posteriors. Save each `.core` file in its own directory labelled `psr_x` where `x` is the index of the pulsar. You'll also have to provide a list of pulsar names which corresponds to the pulsar index.

##### PTA free spectrum data processing


```python
# lets create that density representation using the PTA free spec chain
# first initialise the function by loading the data
kdes = densities.DE_factory(coredir='../data/sim51/freespec_sim51/',
                            recursive=False, pulsar_names=['freespec'])
```

    Loading data from HDF5 file....

we now setup our KDEs are predict probabilities across a grid of log10rho values. All you really need to do is supply an output file location...


```python
kdes.setup_densities(outdir='../data/sim51/freespec_sim51/')
```

    Computing densities for psr 0
    removing nansfrom HDF5 file....





    array([[[-36.04365339, -36.04365339, -36.04365339, ..., -36.04365339,
             -36.04365339, -36.04365339],
            [-36.04365339, -36.04365339, -36.04365339, ..., -36.04365339,
             -36.04365339, -36.04365339],
            [-36.04365339, -36.04365339, -36.04365339, ..., -36.04365339,
             -36.04365339, -36.04365339],
            ...,
            [-36.04365339, -36.04365339, -36.04365339, ..., -36.04365339,
             -36.04365339, -36.04365339],
            [-36.04365339, -36.04365339, -36.04365339, ..., -36.04365339,
             -36.04365339, -36.04365339],
            [-36.04365339, -36.04365339, -36.04365339, ..., -36.04365339,
             -36.04365339, -36.04365339]]])



Well... that was easy! Now just supply this directory to `Ceffyl.ceffyl` and you can do any analysis you want!

So, what exactly is `DE_Factory.setup_densities` doing?

For each posterior file it is supplied, it first calculates the optimal `bandwidth` for a KDE kernel given the data at each frequency. It uses the Sheather-Jones Plug-in Selector which can be found in [`densities.bw.sj`](../ceffyl/bw/). Other bandwidth options/values can be selected, but we find the Sheather-Jones method to be robust for KDE representation of our posteriors.

This step can be slow, depending on how many data points are supplied. If many single pulsar free spectra are being represented, we recommend calculating the bandwidths of the posteriors in parallel as a separate script, saving those values, and supplying them as an ($N_p \times N_f$)-array into the `setup_densities` method.

I.e. if you have an array of single pulsar free spectrum posteriors
- take one pulsar posterior
- save it as a `.core` file
- initialise the `DE_Factory` function with the relevent `.core` file
- use `DE_Factory.bandwidth` to calculate the bandwidths at each frequency
- save the bandwidth array
- do this for all pulsars in parallel
- then reload the `DE_Factory` function with all required free spectrum posteriors, and load the bandwidths into an ($N_p \times N_f$)-array, and supply into `setup_densities`. This is much faster!

What does `setup_densities` do with the bandwidths? It supplies it to the `density` method which creates a `KDEpy` object. `KDEpy` is our kernel density estimator object. By default, we select an Epanechnikov kernel, and supply the data. Once the KDE is created, we calculate probability densities across a grid of $\log_{10}\rho$. We recommend that the the minimum and maximum of the grid be equal to the extrema of the prior space of the free spectrum analysis, and that the spacings between the grid points be less than the bandwidths of the KDEs.

We then save the density file. Other metadata is also saved. And that's it! Your data is now represented and ready to be refit!

Find an example to calculate just the bandwidths at the end of the [single pulsar free spectrum script!](./lamb_2023/scripts/single_psr_fs.py)


