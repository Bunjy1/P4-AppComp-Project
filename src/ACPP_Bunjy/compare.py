""" Complete fit_compare function to apply processes designed in tfp and decomp files.
    Represents summative work of the project.
    
    Each function includes a description along with required inputs and outputs.
    Examples of usage can be found in the related example notebook.
"""

""" Importing Required Dependencies."""
import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy import optimize
from tqdm import tqdm

""" Extracting Required functions from other .py files. """
import ACPP_Bunjy.tfp as tfp
import ACPP_Bunjy.decomp as decomp

""" Function """
def fit_compare(dataset, vectors, params_guess, bounds, algorithm='SVD', metrics=False, component_plots=False, param_plots=False, residuals=False, iterative_fitting=True, clean_outliers=False, save_data=False, filename='placeholder_name'):
    """ Cumulative function to apply both the primary decomposition and TFP processes
        in a single action, with various toggle to control aspects of the process.
        With a correctly processed dataset using the auto_import function in decomp.py,
        this function should be able to achieve the majority of the project's objectives
        in a single run.

        Required Parameters
        -------------------
        dataset: hs.signals.Signal1D
            Imported and processed dataset, optimally the output of decomp.auto_import.
        vectors: int
            The number of desried vector components for reconstruction after the decomposition.
        params_guess: list
            Initial guess values for the TFP fitted parameters.
        bounds: list or array
            Paired arrays defining the lower and upper bounds of possible parameters respectively.

        Optional Parameters
        -------------------
        algorithm: str
            Algorithm for use by Hyperspy decomposition, see Hyperspy documentation.
        metrics: bool, optional
            Toggle to return decomposition goodness metrics.
        component_plots: bool, optional
            Toggle to plot an example set of averaged vector components for visualisation.
        param_plots: bool, optional
            Toggle to plot heatmaps of each found fitted parameter
        residuals: bool, optional
            Toggle to plot residual plots of the parameter heatmaps.
        iterative_fitting: bool, optional
            Toggle for iterative parameter guessing in perfit_iterate, enabled as default.
        clean_outliers: bool, optional
            Toggle to clean outlying data points from error data.
        save_data: bool, optional
            Toggle to save results of the function, useful for big datasets.
        filename: str
            Filename under which to save dataset if save_data=True.

        Returns
        -------
        full_data: dict
            Compiled results of all sub-functions, containing data, parameters, parameter errors, and R^2 array,
            for both raw and clean data, into a dict object with corresponding tage for each, 
            e.g. full_data["Raw"]["R2"] would return the raw R2 array. This is compatible with np.save providing 
            that the data is loaded with allow_pickle=True in np.load, and as data.item().get("Raw","R2").
        """
    # Input error checks
    if not isinstance(dataset, hs.signals.Signal1D):
        raise ValueError("Input data must by of type hs.signals.Signal1D.")
    # All other input types will be flagged by respective function errors.
    
    # Finding vector decomps
    raw_data = dataset.data
    clean_hsdata = decomp.auto_decomp(dataset, vectors, algorithm=algorithm, metrics=metrics)
    clean_data = clean_hsdata.data

    # Plotting components
    if component_plots:
    
        # Average over the selected area and normalize
        area = slice(0, 5) 
        plt.figure(figsize=(8,6))
        plt.title('Raw Electron Scattering Data with Angular Variance, \n Averaged over a 5x5 Pixel Area')
        plt.xlabel('Angle of Rotation (\u00B0)')
        plt.ylabel('Arbitrary Signal Intensity')
        plt.grid()
        raw_avg = np.mean(raw_data[area, area, :], axis=(0, 1))
        raw_norm = decomp.normalize_intensity(raw_avg)
        
        plt.plot(raw_avg, label='raw data (averaged & normalized)')
        plt.show()

        plt.figure(figsize=(6,12))
        for n in range(vectors):
            plt.subplot(vectors+1,1,n+2)
            plt.title('Decomposed Data Vector Component {}'.format(n+1))
            plt.xlabel('Angle of Rotation (\u00B0)')
            plt.ylabel('Arbitrary \n Signal Intensity')
            plt.grid()
            comp_plot = clean_hsdata.get_decomposition_model([n]).data
            comp_avg = np.mean(comp_plot[area, area, :], axis=(0, 1))
            #comp_norm = normalize_intensity(comp_avg)
            plt.plot(comp_avg, label='component {}'.format(n+1))
    
        #plt.legend()
        plt.tight_layout()
        plt.show()

    # Finding fits
    raw_fit = tfp.perfit_iterate(raw_data, params_guess, bounds, iterative_fitting=iterative_fitting)
    clean_fit = tfp.perfit_iterate(clean_data, params_guess, bounds, iterative_fitting=iterative_fitting)

    # Setting paramaters
    raw_params, raw_errors, raw_r2 = raw_fit
    clean_params, clean_errors, clean_r2 = clean_fit

    # Adjusting errors to exclude overly large values
    if clean_outliers == True:
        decomp.outlier_cleaning(raw_errors)
        decomp.outlier_cleaning(clean_errors)

    # Params plots
    if param_plots == True:
        tfp.param_plotting(raw_params, clean_params, 'Parameter', residuals=residuals)
        tfp.param_plotting(raw_errors, clean_errors, 'Error', residuals=residuals)

    # Compiling all results into a singular array
    full_data = {"Raw":{"Data":raw_data, "Params":raw_params, "Errors":raw_errors, "R2":raw_r2},
                 "Clean":{"Data":clean_data, "Params":clean_params, "Errors":clean_errors, "R2":clean_r2}} 
    
    # Saving data into a .npy file
    if save_data == True:
        np.save(filename+'.npy', full_data)

    return(full_data)
