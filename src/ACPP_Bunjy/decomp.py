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

# For git upload eventually, include package definitions with function

def auto_decomp(input_data, vectors, algorithm="SVD", metrics=False, plots=False):

    # Function to automate decompostion functions, with the ability to choose number of reconstruction vectors and algorithm used.
    # Can calculate metrics to evaluate goodness of cleanup, and plot the reconstruction and scree plot
    # the input_data array must be a dataset already processed through hs.load (may change)

    # Decomposition
    input_data.decomposition(algorithm=algorithm) # some algorithms need output_dimension specified?
    cleaned_data = input_data.get_decomposition_model(vectors)

    # Returning metrics
    if metrics == True:
        orig = input_data.data
        clean = cleaned_data.data

        # Relative reconstruction error
        rel_error = np.linalg.norm(orig - clean) / np.linalg.norm(orig)
        print('Relative reconstruction error: {}'.format(rel_error))

        # Explained variance ratio
        evr_raw = cleaned_data.get_explained_variance_ratio().data
        evr = evr_raw[:vectors].sum() # change value from 5 when implementing vector selection
        print('Explained variance ratio: {}'.format(evr))

        # Noise suppresion metric
        orig_std = np.std(orig)
        clean_std = np.std(clean)
        nsr = clean_std / orig_std
        print('Noise suppression metric: {}'.format(nsr))
    
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

