""" Functions associated with plotting the two-fold periodic (TFP) function
    used in the main project section.
    
    Each function includes a description along with required inputs and outputs.
    Examples of usage can be found in the related example notebook.
"""

""" Importing Required Dependencies."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def periodic(x, a, c, d, f, g):
    """ Defining the TFP for fitting to the electron lattice data.
        The equation used is a generic periodic term of the form 
        sin(x) + sin(2x), with paramaters that can be manually toggled
        or curve-fitted.

    Parameters
    ----------
    x: array-like
        Input values of x to the function.
    a,b,c,d,e: int or float
        Parameters representing amplitude, phase and offset of the function.

    Returns
    -------
    y : array-like
        The calculated y values of the periodic function corresponding to x.
    """
    return(a*np.sin(np.deg2rad(x+c))+d*np.sin(np.deg2rad(2*(x+f)))+g)

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
        Paired arrays defining the upper and lower bounds of possible parameters.

    Returns
    -------
    params: array
        Found parameters from curve fitting.
    errors: array
        Associated parameters on each parameter.
    r_squared: float
        R² value associated with the fitted curve.
    """
    deg = np.arange(len(current_data))
    
    params, params_covariance = optimize.curve_fit(periodic, deg, current_data, p0=params_guess, bounds=bounds, maxfev=30000)
    fitted_curve = periodic(deg, *params)
    
    errors = np.sqrt(np.diag(params_covariance))

    residuals = fitted_curve-current_data
    ss_res = sum(residuals**2)
    ss_tot = sum((current_data-np.mean(current_data))**2)
    r_squared = 1 - ss_res/ss_tot

    return(params, errors, r_squared)


def perfit_iterate(inp_array, params_guess, bounds, iterative_fitting=True):
    """ Applies the periodic_fitter function iteratively to a 3D data array.
        Optionally, iterative parameter guessing can be implemented to improve
        function efficiency.

    Parameters
    ----------
    inp_array: array-like
        3D data array for curve fitting to the third array dimension.
    params_guess: list or array
        Array of initial estimate parameters for function values a-e.
    bounds: list or array
        Paired arrays defining the upper and lower bounds of possible parameters.
    iterative_fitting: bool, optional
        Argument toggle for implementing the iterative parameter guessing
        by using the parameter set found for the previous point in the iteration.
        
    Returns
    -------
    params_array: array
        Stacked array of fitted parameters 
    errors_array: array
        Stacked array of errors associated to found parameters.
    r2_array: array
        List of R² values.
    """
    
    params_array = []
    errors_array = []
    r2_array = []
    
    total_iters = inp_array.shape[0] * inp_array.shape[1]
    
    with tqdm(total=total_iters, desc="Fitting periodic functions") as pbar:
        for n in range(inp_array.shape[0]):
            for i in range(inp_array.shape[1]):
                
                current_data = inp_array[n, i, :]
                if iterative_fitting == True:
                    if n == 0 and i == 0:
                        fit = periodic_fitter(current_data, params_guess, bounds)
                    else:
                        fit = periodic_fitter(current_data, current_params, bounds)
                else:
                    fit = periodic_fitter(current_data, params_guess, bounds)
                    
                
                params = fit[0]
                errors = fit[1]
                r_squared = fit[2]
                current_params = params
                
                params_array.append(params)
                errors_array.append(errors)
                r2_array.append(r_squared)
                
                pbar.update(1)

    params_array = np.stack(params_array)
    errors_array = np.stack(errors_array)
    r2_array = np.stack(r2_array)
    
    return params_array, errors_array, r2_array

def param_plotting(dataset, raw_pdata, clean_pdata, index, name):
    plt.figure(figsize=(10,4))
    plt.suptitle(name+' Set {} Colourmap'.format(index+1))
            
    plt.subplot(1,2,1)
    plt.title('Raw Values')
    raw_pshape = raw_pdata[:,index].reshape(dataset.shape[0],dataset.shape[1])
    plt.imshow(raw_pshape)
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title('Clean Values')
    clean_pshape = clean_pdata[:,index].reshape(dataset.shape[0],dataset.shape[1])
    plt.imshow(clean_pshape)
    plt.colorbar()
