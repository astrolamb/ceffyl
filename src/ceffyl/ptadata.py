"""

"""

import numpy as np
from numpy.typing import NDArray
import json

class PTAData:
    """
    A class to store information about a pulsar timing array (PTA).
    The PTAData class is designed to store the pulsar timing array data,
    including the pulsar names, frequencies, log densities, density grid,
    and time span. It provides methods to retrieve the pulsar name, frequencies,
    log density, density grid, and time span.
    It is used to represent the data associated with a pulsar in a pulsar timing
    array, including the frequencies at which the pulsar is observed, the
    log density of the pulsar data, the density grid, and the time span of the
    pulsar data.

    Attributes:
        name: The name of the pulsar.
        freqs: An instance of the Frequencies class containing the frequencies
               and their corresponding indices.
        log_density: A (n_freq, n_grid) array where n_freq is the number of
                     frequencies and n_grid is the number of bins in the density
                     grid, representing the log density of the pulsar data.
        density_grid: A (n_grid,) array where n_grid is the number of bins in
                      the density grid, representing the density grid of the pulsar data.
        tspan: The time span of the pulsar data.

    """
    def __init__(self,
                 pulsar_names: list[str],
                 freqs: NDArray[np.float64],
                 log_densities: NDArray[np.float64],
                 density_grid: NDArray[np.float64],
                 param_labels: list[str],
                 tspan: NDArray[np.float64],
                 chain_processing_details: dict = None):
        """
        Initialise a Pulsar object with information about a pulsar.

        Args:
            name:
                The name of the pulsar.
            freqs:
                The frequencies of the pulsar data. This is an instance of the
                Frequencies class, which contains the frequencies and their
                corresponding indices.
            log_density:
                The log_density of the pulsar data. This is a (n_psr, n_freq, n_grid)
                array where n_psr is the number of pulsars, n_freq is the number of frequencies, and n_grid is
                the number of bins in the density grid.
            density_grid:
                The density grid of the pulsar data. This is a (n_grid,) array
                where n_grid is the number of bins in the density grid.
            tspan:
                The time span of the pulsar data.
        """

        if len(param_labels) != len(freqs):
            raise ValueError("Length of param_labels must match the number of frequencies.")
        
        if log_densities.shape[0] != len(pulsar_names):
            raise ValueError("Number of pulsars in log_densities must match the length of pulsar_names.")
        if log_densities.shape[1] != len(freqs):
            raise ValueError("Number of frequencies in log_densities must match the length of freqs.")
        if density_grid.ndim != 1:
            raise ValueError("density_grid must be a 1D array.")

        self.pulsar_names = pulsar_names
        self.freqs = freqs
        self.log_densities = log_densities
        self.density_grid = density_grid
        self.param_labels = param_labels
        self.tspan = tspan
        if chain_processing_details is None:
            self.chain_processing_details = {}
        else:
            self.chain_processing_details = chain_processing_details 
    
    def __repr__(self):
        return (f"PTAData(pulsar_names={self.pulsar_names}, "
                f"freqs={self.freqs}, "
                f"log_densities={self.log_densities}, "
                f"density_grid={self.density_grid}, "
                f"tspan={self.tspan})")
    
    def __str__(self):
        return (f"PTAData with {len(self.pulsar_names)} pulsars, "
                f"{self.num_frequencies()} frequencies, "
                f"log densities shape: {self.log_densities.shape}, "
                f"density grid shape: {self.density_grid.shape}, "
                f"time span: {self.tspan}")

    def get_pulsar_names(self) -> str:
        """Return the names of the pulsars."""
        return self.pulsar_names
    
    def get_frequencies(self) -> NDArray[np.float64]:
        """Return the frequencies of the pulsar."""
        return self.freqs
    
    def get_freqs_from_indices(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Return the frequencies corresponding to the given indices.

        Args:
            indices: An array of indices for which to return the frequencies.

        Returns:
            An array of frequencies corresponding to the given indices.
        """
        return self.freqs[indices]
    
    def num_frequencies(self) -> int:
        """Return the number of frequencies."""
        return len(self.freqs)
    
    def get_logpdf(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Return the log density for the given indices.

        Args:
            indices: An array of indices for which to return the log density.

        Returns:
            An array of log densities corresponding to the given indices.
        """
        return self.log_densities[indices]
    
    def get_density_grid(self) -> NDArray[np.float64]:
        """
        Return the density grid of the pulsar.

        Returns:
            An array representing the density grid of the pulsar.
        """
        if self.density_grid is None:
            raise ValueError("Density grid is not set.")
        return self.density_grid
    
    # method to save PTAData as a json file
    def save_as_json(self, filename: str):
        """
        Save the PTAData object as a JSON file.

        Args:
            filename: The name of the file to save the data to.
        """
        data = {
            "pulsar_names": self.pulsar_names,
            "freqs": self.freqs.tolist(),
            "log_densities": self.log_densities.tolist(),
            "density_grid": self.density_grid.tolist(),
            "param_labels": self.param_labels,
            "tspan": self.tspan,
            "chain_processing_details": self.chain_processing_details
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def from_json(cls, filename: str):
        """
        Load a PTAData object from a JSON file.

        Args:
            filename: The name of the file to load the data from.

        Returns:
            An instance of the PTAData class.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls(
            pulsar_names=data["pulsar_names"],
            freqs=np.array(data["freqs"]),
            log_densities=np.array(data["log_densities"]),
            density_grid=np.array(data["density_grid"]),
            param_labels=data.get("param_labels", []),
            tspan=np.array(data["tspan"]),
            chain_processing_details=data['chain_processing_details']
        )
