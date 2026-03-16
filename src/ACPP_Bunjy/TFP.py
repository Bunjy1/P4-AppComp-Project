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

        """
    return(a*np.sin(np.deg2rad(x+c))+d*np.sin(np.deg2rad(2*(x+f)))+g)

def periodic_fitter(inp_array, params_guess):

    # This function takes an input 3D data array and iterates over it to generate fitted periodic functions to the intensity pattern, it also takes an array of input
    # guess values for the curve fitting. It returns an array of the found paramaters for each iteration of the fitting and an associated R-squared goodness of fit.
    
    params_array = []
    r2_array = []
    run = 1

    def periodic(x, a, b, c, d, e, f, g):
        return(a*np.sin(b*np.deg2rad(x)+c)+d*np.cos(e*np.deg2rad(x)+f)+g)
        #return(a*np.sin(b*np.deg2rad(x)+c)**2+g)
    
    for n in range(inp_array.shape[0]):
        for i in range(inp_array.shape[1]):
            current_data = inp_array[n,i,:]
            deg = np.arange(0,len(current_data), 1)
            
            params, params_covariance = optimize.curve_fit(periodic, deg, current_data, p0=params_guess, maxfev=3000)
            fitted_curve = periodic(deg, *params)
            
            errors = np.sqrt(np.diag(params_covariance))

            residuals = fitted_curve-current_data
            ss_res = sum(residuals**2)
            ss_tot = sum((current_data-np.mean(current_data))**2)
            r_squared = 1 - ss_res/ss_tot

            params_array.append(params)
            r2_array.append(r_squared)

            #plt.plot(params)
            #plt.scatter(run, params[0])
            #print('Run {}'.format(run))
            run = run+1
    params_array = np.stack(params_array)
    return(params_array, r2_array)
