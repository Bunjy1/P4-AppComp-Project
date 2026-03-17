""" Primary function associated with applying UML decomposition to a dataset.
    Additional functions for calculation of metrics of goodness, and 
    standardizing of dataset where necessary can also be found here.
    
    Each function includes a description along with required inputs and outputs.
    Examples of usage can be found in the related example notebook.
"""

""" Importing Required Dependencies."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm
import hyperspy.api as hs

""" Functions """
def rre(orig, clean):
    """ Function to find the relative reconstruction error of a
        decomposition algorithm. Typically used within the later
        auto_decomp function, but can be applied manually.

        Parameters
        ----------
        orig: numpy.nparray
            Raw dataset before decomposition.
        clean: numpy.nparray
            Cleaned dataset after decomposition and reconstruction.

        Returns
        -------
        rel_error: float
            Relative reconstruction error of decomposition, see example notebook for details.
        """
    rel_error = np.linalg.norm(orig - clean) / np.linalg.norm(orig)
    print('Relative reconstruction error: {}'.format(rel_error))
    return(rel_error)

def evr(orig, clean):
    """ Function to find the explained variance ratio of a
        decomposition algorithm. Typically used within the later
        auto_decomp function, but can be applied manually.

        Parameters
        ----------
        orig: numpy.nparray
            Raw dataset before decomposition.
        clean: numpy.nparray
            Cleaned dataset after decomposition and reconstruction.

        Returns
        -------
        evr: float
            Explained variance ratio of decomposition, see example notebook for details.
        """
    evr_raw = cleaned_data.get_explained_variance_ratio().data
    evr = evr_raw[:vectors].sum() # change value from 5 when implementing vector selection
    print('Explained variance ratio: {}'.format(evr))
    return(evr)

def nsm(orig, clean):
    """ Function to find the noise suppression metric of a
        decomposition algorithm. Typically used within the later
        auto_decomp function, but can be applied manually.

        Parameters
        ----------
        orig: numpy.nparray
            Raw dataset before decomposition.
        clean: numpy.nparray
            Cleaned dataset after decomposition and reconstruction.

        Returns
        -------
        nsr: float
            Noise suppression metric of decomposition, see example notebook for details.
        """
    orig_std = np.std(orig)
    clean_std = np.std(clean)
    nsr = clean_std / orig_std
    print('Noise suppression metric: {}'.format(nsr))
    return(nsr)
    
def auto_decomp(input_data, vectors, algorithm="SVD", metrics=False, plots=False):

    """ Function to automate and streamline Hyperspy decomposition
        of input data. Data can be reconstructed with a desired number
        of vectors, different UML algorithms, and otional returns of
        the decomposition function plots and metrics relating to
        goodness of fit.

        Parameters
        ----------
        input_data: hyperspy.signals.Signal1D
            Input raw data for decomposition, formatted in Hyperspy signal form.
        vectors: int
            Desired vectors to be used for signal reconstruction, for which all vectors up to the int value will be used.
        algorithm: str, default='SVD'
            The algorithm to be used by Hyperspy for decomposition, see Hyperspy documentation for details.
        metrics: bool, optional
            Argument to return metrics for evaluation of algorithm performance, including RRE, EVR, and NSM.
        plots: bool, optional
            Argument to plot decomposition and scree plot in the Hyperspy framework, again see HS documentation.

        Returns
        -------
        Cleaned data: hyperspy.signals.Signal1D
            Reconstructed data using set vectors, with noise removed.
        """

    # Decomposition
    input_data.decomposition(algorithm=algorithm) # some algorithms need output_dimension specified?
    cleaned_data = input_data.get_decomposition_model(vectors)

    # Returning metrics
    if metrics == True:
        orig = input_data.data
        clean = cleaned_data.data

        

        

        
    
    # Finding recommended number of decomp vectors?
    # Base on an analysis of metrics and scree plot?

    # Optional plotting
    if plots == True:
        # Decomp plot
        cleaned_data.plot()
        # Scree plot
        cleaned_data.plot_explained_variance_ratio()

    return(cleaned_data)

def outlier_cleaning(errors):
    for n in range(errors.shape[1]):
        col = errors[:, n]
        mean = np.mean(col)
        std = np.std(col)
    
        upper = mean + 3*std
        lower = mean - 3*std
        errors[:, n] = np.clip(col, lower, upper)

  # Function to normalize a 1D array between 0 and 1
def normalize_intensity(spectrum):
    return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

