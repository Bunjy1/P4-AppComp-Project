""" Primary function associated with applying UML decomposition to a dataset.
    Additional functions for calculation of metrics of goodness, and 
    standardizing of dataset where necessary can also be found here.
    
    Each function includes a description along with required inputs and outputs.
    Examples of usage can be found in the related example notebook.
"""

""" Importing Required Dependencies."""
import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

""" Functions """
def auto_import(filename, crop=None):
    """ Automated function for importing of datasets to be processed.
        Optionally, data can be cropped to a smaller size, useful for
        testing functions.

        Parameters
        ----------
        filename: str
            Name of the file to be imported, not including directory path
        crop : tuple of slices, optional
            Cropping instructions, e.g. (slice(0,10), slice(0,10), slice(None)) will return a crop of [0:10,0:10,:].
            Function will not crop and return full dataset as default.

        Returns
        -------
        cropped_data: hs.signals.Signal1D
            Correctly unpackaged raw data of desired size.
        """

    # Error check if filename is a str
    if not isinstance(filename, str):
        raise TypeError(f"filename must be a string, got {type(filename)}")

    # Create full filename with path and check it exists
    filepath = os.getcwd()+filename
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    # Load data and convert to np array for cropping
    dataset = hs.load(filepath)
    np_dataset = dataset.data

    # Optional cropping with error checks on inputs
    if crop is not None:
        # Checks for correct formatting of crop argument
        if not isinstance(crop, tuple):
            raise TypeError("crop must be a tuple of slice objects.")

        # Checks crop argument is of same dimension as the dataset
        if len(crop) != np_dataset.ndim:
            raise ValueError(f"crop must have {np_dataset.ndim} dimensions, got {len(crop)}.")

        # Checks that desired crop dimension does not exceed total dimension of the dataset
        for i, s in enumerate(crop):
            if s.stop and s.stop > np_dataset.shape[i]:
                raise ValueError(f"Crop exceeds dimension {i}: max={np_dataset.shape[i]}, got stop={s.stop}")

        # Cropping data if crop is correctly formatted
        np_dataset = np_dataset[crop]

    # Converting data back to hs format for use in auto_decomp function
    cropped_data = hs.signals.Signal1D(np_dataset)
    return(cropped_data)
        
def find_rre(orig, clean):
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

    # Checking formatting of inputs to return errors if incorrect
    if not all(isinstance(v, np.ndarray) for v in (orig,clean)):
        raise ValueError("Raw and clean data arrays must by of type numpy.ndarray.")

    # Calculating RRE
    rel_error = np.linalg.norm(orig - clean) / np.linalg.norm(orig)
    print('Relative reconstruction error: {}'.format(rel_error))
    return(rel_error)

def find_evr(cleaned_data, vectors, algorithm):
    """ Function to find the explained variance ratio of a
        decomposition algorithm. Typically used within the later
        auto_decomp function, but can be applied manually.
        
        Note: EVR calculation is only available for SVD and PCA
        decompositions.

        Parameters
        ----------
        cleaned_data: hs.signals.Signal1D
            Cleaned dataset after decomposition and reconstruction in Hyperspy formatting.
        vectors: int
            Number of vectors in cleaned_data reconstruction.
        algorithm: str
            Algorithm used for decomposition of the dataset.

        Returns
        -------
        evr: float
            Explained variance ratio of decomposition, see example notebook for details.
        """

    # Checking formatting of inputs to return errors if incorrect
    if not isinstance(cleaned_data, hs.signals.Signal1D):
        raise ValueError("Raw and clean data arrays must by of type hs.signals.Signal1D.")

    # Checking correct algorithm for calculation
    if algorithm not in ["SVD","sklearn_pca"]:
        print("EVR Metric is unavailable for", algorithm)
        return None

    # Calculating EVR
    evr_raw = cleaned_data.get_explained_variance_ratio().data
    evr = evr_raw[:vectors].sum() # change value from 5 when implementing vector selection
    print('Explained variance ratio: {}'.format(evr))
    return(evr)

def find_nsm(orig, clean):
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

    # Checking formatting of inputs to return errors if incorrect
    if not all(isinstance(v, np.ndarray) for v in (orig,clean)):
        raise ValueError("Raw and clean data arrays must by of type numpy.ndarray.")

    # Calculating NSM
    orig_std = np.std(orig)
    clean_std = np.std(clean)
    nsm = clean_std / orig_std
    print('Noise suppression metric: {}'.format(nsm))
    return(nsm)
    
def auto_decomp(input_data, vectors, algorithm="SVD", output_dimension=None, metrics=False, plots=False):

    """ Function to automate and streamline Hyperspy decomposition
        of input data. Data can be reconstructed with a desired number
        of vectors, different UML algorithms, and otional returns of
        the decomposition function plots and metrics relating to
        goodness of fit.

        Note: Scree plots are only available for SVD and PCA
        decompositions.

        Parameters
        ----------
        input_data: hs.signals.Signal1D
            Input raw data for decomposition, formatted in Hyperspy signal form.
        vectors: int
            Desired vectors to be used for signal reconstruction, for which all vectors up to the int value will be used.
        algorithm: str, default='SVD'
            The algorithm to be used by Hyperspy for decomposition, see Hyperspy documentation for details.
        output_dimension: int, default='None'
            The number of output components desired from decomposition, required for operation of other non-SVD algorithms in
            Hyperspy. None as default for use with SVD, which will give an output_dimension of 360.
        metrics: bool, optional
            Argument to return metrics for evaluation of algorithm performance, including RRE, EVR, and NSM.
        plots: bool, optional
            Argument to plot decomposition and scree plot in the Hyperspy framework, again see HS documentation.

        Returns
        -------
        cleaned_data: hs.signals.Signal1D
            Reconstructed data using set vectors, with noise removed.
        metrics_list: list
            List of the three calculated goodness metrics.
        """

    # Checking formatting of inputs to return an error if incorrect
    if not isinstance(input_data, hs.signals.Signal1D):
        raise ValueError("Input data must by of type hs.signals.Signal1D.")
        
    if not isinstance(vectors, int):
        raise ValueError("Number of reconstruction vectors must be an integer value.")

    # Hyperspy algorithm will return separate error if an incorrect algorithm name is used so unneeded here.
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm must be a str. See Hyperspy documentation for list of options.")

    # Decomposition
    input_data.decomposition(algorithm=algorithm, output_dimension=output_dimension)
    cleaned_data = input_data.get_decomposition_model(vectors)

    # Setting metrics_list empty if unneeded
    metrics_list = None
    
    # Returning metrics
    if metrics:
        # Redefining datasets into nparrays
        orig = input_data.data
        clean = cleaned_data.data

        # Running metric functions
        rre = find_rre(orig, clean)
        evr = find_evr(cleaned_data, vectors, algorithm=algorithm)
        nsm = find_nsm(orig,clean)
        metrics_list = [rre, evr, nsm]

    # Optional plotting
    if plots:
        # Decomp plot
        cleaned_data.plot()

        # Scree plot
        # Checking a scree plot compatible algorithm is being used
        if algorithm not in ["SVD","sklearn_pca"]:
            print("Scree Plot is unavailable for", algorithm)
        else:
            cleaned_data.plot_explained_variance_ratio()

    return(cleaned_data, metrics_list)

def outlier_cleaning(data):
    """ Function to locate and clean data points of extreme value
        from parameter and error plots, helpful in normalising
        plots.

        Parameter
        ---------
        data: numpy.nparray
            Data for cleaning

        Returns
        -------
        Modifies original input data to the cleaned dataset.
        No direct function returns.
        """

    # Checking formatting of input to return an error if incorrect
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must by of type numpy.ndarray.")

    # Iterating over 1D columns of the dataset for ease of calculation
    for n in range(data.shape[1]):
        # For each column, the mean and std values are calculated
        col = data[:, n]
        mean = np.mean(col)
        std = np.std(col)

        # Values determined to be outwith 3 standard deviations from the mean are cleaned
        # Standard method for identification of outliers, found points are clipped to the
        # limits established at p/m 3 std.
        upper = mean + 3*std
        lower = mean - 3*std
        data[:, n] = np.clip(col, lower, upper)


def normalize_intensity(data):
    """ Utility function to normalize a spectrum to values
        between 1 and 0. Helpful for comparative plotting.

        Parameter
        ---------
        data: numpy.nparray
            Data to be normalized.

        Returns
        -------
        data_norm: nump.nparray
            Normalized dataset.
        """
    
    # Checking formatting of input to return an error if incorrect
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must by of type numpy.ndarray.")

    # Normalizing data
    data_norm = (data - np.min(data))/(np.max(data) - np.min(data))
    return(data_norm)

