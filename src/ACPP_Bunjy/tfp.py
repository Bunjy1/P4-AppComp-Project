""" Functions associated with plotting the two-fold periodic (TFP) function
    used in the main project section.
    
    Each function includes a description along with required inputs and outputs.
    Examples of usage can be found in the related example notebook.
"""

""" Importing Required Dependencies."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm

""" Functions """

def periodic(x, a, b, c, d, e):
    """ Defining the TFP for fitting to the electron lattice data.
        The equation used is a generic periodic term of the form 
        sin(x) + sin(2x), with parameters that can be manually adjusted
        or determined through curve fitting.

    Parameters
    ----------
    x: array-like
        Input values of x to the function (in degrees).
    a: float
        Amplitude of the first sinusoidal term.
    b: float
        Phase shift applied to the first sinusoidal term.
    c: float
        Amplitude of the second sinusoidal term.
    d: float
        Phase shift applied to the second sinusoidal term.
    e: float
        Constant offset applied to the function.

    Returns
    -------
    y : numpy.ndarray
        The calculated y values of the periodic function corresponding to x.
    """
    x = np.asarray(x)
    return(a*np.sin(np.deg2rad(x+b)) + c*np.sin(np.deg2rad(2*x+d)) + e)

def periodic_fitter(current_data, params_guess, bounds):
    """ Application of scipy's optimise curve-fit to the TFP in a
        generic singular case.

    Parameters
    ----------
    current_data: array-like
        Intensity of y-values for the data being fitted to.
    params_guess: list or array
        Array of initial estimate parameters for function values a-e.
    bounds: list or array
        Paired arrays defining the lower and upper bounds of possible parameters respectively.

    Returns
    -------
    params: numpy.ndarray
        Found parameters from curve fitting.
    errors: numpy.ndarray
        Associated parameters on each parameter.
    r_squared: float
        R² value associated with the fitted curve.
    """

    # Converting input data to numpy array (if not already)
    current_data = np.asarray(current_data)

    # Checking all inputs are of correct type and returning errors if not
    if current_data.ndim != 1:
        raise ValueError("current_data must be a one-dimensional array of y-values.")

    if len(current_data) == 0:
        raise ValueError("current_data cannot be empty.")

    if len(params_guess) != 5:
        raise ValueError("params_guess must contain exactly 5 parameters: [a, b, c, d, e].")

    if len(bounds) != 2:
        raise ValueError("bounds must contain two arrays: (lower_bounds, upper_bounds).")

    # Defining bounds from input
    lower_bounds, upper_bounds = bounds

    if len(lower_bounds) != 5 or len(upper_bounds) != 5:
        raise ValueError("Both lower_bounds and upper_bounds must contain 5 values.")

    if np.any(np.asarray(lower_bounds) >= np.asarray(upper_bounds)):
        raise ValueError("Each lower bound must be strictly less than the corresponding upper bound.")

    # Running curve-fit optimize 
    deg = np.arange(len(current_data))
    params, params_covariance = optimize.curve_fit(periodic, deg, current_data, p0=params_guess, bounds=bounds, maxfev=30000)
    
    # Determining curve from found parameters and associated errors
    fitted_curve = periodic(deg, *params)
    errors = np.sqrt(np.diag(params_covariance))

    # Finding R² value
    residuals = fitted_curve - current_data
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((current_data - np.mean(current_data))**2)
    r_squared = 1 - ss_res / ss_tot

    return(params, errors, r_squared)

def perfit_iterate(inp_array, params_guess, bounds, iterative_fitting=True):
    """ Applies the periodic_fitter function iteratively to a 3D data array.
        Optionally, iterative parameter guessing can be implemented to 
        improve function efficiency.

    Parameters
    ----------
    inp_array: array-like
        3D data array for curve fitting along the third array dimension.
    params_guess: list or array-like
        Initial parameter guesses for the periodic function [a, b, c, d, e].
    bounds: tuple of array-like
        Paired arrays defining the lower and upper bounds of possible parameters.
    iterative_fitting: bool, optional
        Toggle for iterative parameter guessing using the previous fit parameters.

    Returns
    -------
    params_array: numpy.ndarray
        Array of fitted parameters with shape (nx, ny, 5).
    errors_array: numpy.ndarray
        Array of parameter uncertainties with shape (nx, ny, 5).
    r2_array: numpy.ndarray
        Array of R² values with shape (nx, ny).
    """

    # Convert input to numpy array (if not already)
    inp_array = np.asarray(inp_array)

    # Checking input formatting and returning errors if incorrect
    if inp_array.ndim != 3:
        raise ValueError("inp_array must be a 3D array.")

    if len(params_guess) != 5:
        raise ValueError("params_guess must contain 5 parameters: [a, b, c, d, e].")

    # Setting data shape variables and creating preshaped arrays for calculated values
    nx, ny, _ = inp_array.shape
    params_array = np.zeros((nx, ny, 5))
    errors_array = np.zeros((nx, ny, 5))
    r2_array = np.zeros((nx, ny))
    total_iters = nx * ny

    # Running iterative function with tdqm to allow for progress bar
    with tqdm(total=total_iters, desc="Fitting periodic functions") as pbar:
        for n in range(nx):
            for i in range(ny):

                current_data = inp_array[n, i, :]

                # Iterative plotting conditional to use previouly found parameters
                # This highly optimises runtime of the function (see report)
                if iterative_fitting:
                    if n == 0 and i == 0:
                        fit = periodic_fitter(current_data, params_guess, bounds)
                    else:
                        fit = periodic_fitter(current_data, current_params, bounds)
                else:
                    fit = periodic_fitter(current_data, params_guess, bounds)

                # Defining found values and appending into empty arrays
                params, errors, r_squared = fit
                current_params = params
                params_array[n, i, :] = params
                errors_array[n, i, :] = errors
                r2_array[n, i] = r_squared

                # Update progress bar
                pbar.update(1)

    return params_array, errors_array, r2_array

def param_plotting(raw_pdata, clean_pdata, name, residuals=False, cmap="viridis"):
    """ Function for plotting the results of the perfit_iterate function,
        showing the compared raw and clean parameter plots of each output
        parameter. Function is designed to operate on two runs of perfit, 
        one raw data fit and one clean fit. Can also be applied to plot the 
        error values.

        Parameters
        ----------
        raw_pdata: numpy.nparray
            Raw data output array from the perfit function, either params_array or errors_array.
        clean_pdata: numpy.nparray
            Clean data output array from the perfit function, either params_array or errors_array.
        name: str
            Name of the value being plotted, typically 'Parameter' or 'Error'.
        residuals: bool, optional
            Optional toggle to plot the residual difference of each plot pair in the subplot.
        cmap: str, optional
            Colourmap to use for imshow function, viridis as default.
            Options for colours: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    """
    # Checking formatting of inputs to return errors if incorrect
    if not all(isinstance(v, np.ndarray) for v in (raw_pdata,clean_pdata)):
        raise ValueError("Raw and clean data arrays must by of type numpy.ndarray.")
        
    if raw_pdata.shape != clean_pdata.shape:
        raise ValueError("Raw and clean data arrays must be the same size.")

    if not isinstance(name, str):
        raise ValueError("Name must be a string.")
        
    # Defining the number of subplots from the residual toggle
    if residuals:
        plot_no = 3
    else:
        plot_no = 2

    # Iterating the plots over the number of parameters in the array.
    for index in range(raw_pdata.shape[2]):

        # Paramaters for title
        p_name = ["A","B","C","D","E"]

        # Formatting plot
        plt.figure(figsize=(12,4))
        plt.suptitle(name+' Set {} Colourmap'.format(p_name[index]))

        # Raw value plot
        plt.subplot(1,plot_no,1)
        plt.title('Raw Values')
        raw_pshape = raw_pdata[:,:,index]
        plt.imshow(raw_pshape, cmap=cmap)
        plt.colorbar()

        # Clean value plot
        plt.subplot(1,plot_no,2)
        plt.title('Clean Values')
        clean_pshape = clean_pdata[:,:,index]
        plt.imshow(clean_pshape, cmap=cmap)
        plt.colorbar()

        # Residuals plot (if needed)
        if residuals:
            plt.subplot(1,plot_no,3)
            plt.title('Residuals')
            residuals_shape = clean_pshape - raw_pshape
            plt.imshow(residuals_shape, cmap=cmap)
            plt.colorbar()
