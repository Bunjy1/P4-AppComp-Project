import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


"""Compiled set of functions developed during the first three weeks of lab sessions.

Each function includes a description along with required inputs and outputs.
Examples of usage can be found in the related example notebook.
"""


def double(input):
    """Return the input value multiplied by two.

    Parameters
    ----------
    input : float or int
        The number to be doubled.

    Returns
    -------
    float or int
        The input value multiplied by 2.
    """
    output = 2 * input
    return output

"" Week 2 Functions ""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

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

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def polynomial_fitter(x_data, y_data, order, params_guess):

    # This function plots a figure of a 2D dataset and fits a polynomial function to it, displaying the R squared value, returning an array of fitted paramaters, and plotting
    # the residuals. The function takes four inputs, the x and y axes of the data, and the desired order of polynomial. x_data and y_data must both be numpy arrays, and 
    # order a float, up to sixth order (input between 1 and 6). Params_guess should be an array defining the guessed input paramaters to the curve fit.

    # Setting up subplots
    plt.figure(figsize=(10,8))
    plt.suptitle('Raw Data and Fitted Line Plot, with Residuals and Line Fit Parameters')
    
    # Designing graph
    plt.subplot(2,1,1)
    plt.scatter(x_data, y_data, marker='x', color='black', label='Raw Data')
    plt.title('Raw Data and Fitted Line Plot')
    
    # Defining Curve and Fitting
    # This could be optimised to generate a new function for a given order input but I didn't have time to get that working
    def order1(x, a, b):
        return(a+b*x)

    def order2(x, a, b, c):
        return(a+b*x+c*x**2)

    def order3(x, a, b, c, d):
        return(a+b*x+c*x**2+d*x**3)

    def order4(x, a, b, c, d, e):
        return(a+b*x+c*x**2+d*x**3+e*x**4)

    def order5(x, a, b, c, d, e, f):
        return(a+b*x+c*x**2+d*x**3+e*x**4+f*x**5)

    def order6(x, a, b, c, d, e, f, g):
        return(a+b*x+c*x**2+d*x**3+e*x**4+f*x**5+g*x**6)

    polynomial = [order1, order2, order3, order4, order5, order6]
    chosen_poly = polynomial[order-1]

    # Fitting curve and defining paramaters for display
    params, params_covariance = optimize.curve_fit(chosen_poly, x_data, y_data, p0=params_guess)
    errors = np.sqrt(np.diag(params_covariance))

    fitted_curve = chosen_poly(x_data, *params)

    # Determining residuals and R^2 value
    residuals = (fitted_curve-y_data)

    ss_res = sum(residuals**2)
    ss_tot = sum((y_data-np.mean(y_data))**2)
    r_squared = 1 - ss_res/ss_tot

    # Plotting curve fit
    plt.plot(x_data, fitted_curve, color='red', label='Best Fitting Line \nR\u00B2={:.4g}'.format( r_squared))
    plt.legend()
    plt.grid()

    # Plotting Residuals
    plt.subplot(2,1,2)
    plt.title('Residuals Plot')
    plt.plot(x_data, residuals, label='Residual Values')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    print(params)
    
    return(params, errors, r_squared)

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def straightline_fitter(x_data, y_data):

    # This function plots a figure of a 2D dataset and fits a linear function to it, displaying the paramaters of the line with errors, the R squared value, and plotting
    # the residuals. The function takes two inputs, the x and y axes of the data. These must both be numpy arrays.

    # Setting up subplots
    plt.figure(figsize=(10,8))
    plt.suptitle('Raw Data and Fitted Line Plot, with Residuals and Line Fit Parameters')
    
    # Designing graph
    plt.subplot(2,1,1)
    plt.scatter(x_data, y_data, marker='x', color='black', label='Raw Data')
    plt.title('Raw Data and Fitted Line Plot')
    
    # Defining Curve and Fitting
    def straight_line(x, m, c):
        return(m*x+c)

    # Fitting curve and defining paramaters for display
    params, params_covariance = optimize.curve_fit(straight_line, x_data, y_data, p0=[1,0])
    grad = params[0]
    y_int = params[1]

    errors = np.sqrt(np.diag(params_covariance))
    grad_err = errors[0]
    y_int_err = errors[1]

    # Determining residuals and R^2 value
    residuals = (straight_line(x_data, grad, y_int)-y_data)

    ss_res = sum(residuals**2)
    ss_tot = sum((y_data-np.mean(y_data))**2)
    r_squared = 1 - ss_res/ss_tot

    # Plotting curve fit
    plt.plot(x_data, straight_line(x_data, grad, y_int), color='red', label='Best Fitting Line \nParameters: \nm={:.5g}\u00B1{:.2g} \nc={:.3g}\u00B1{:.2g} \nR\u00B2={:.4g}'.format(grad, grad_err, y_int, y_int_err, r_squared))
    plt.legend()
    plt.grid()

    # Plotting Residuals
    plt.subplot(2,1,2)
    plt.title('Residuals Plot')
    plt.plot(x_data, residuals, label='Residual Values')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    
    return(params, errors, r_squared)

"" Week 3 Functions ""

### Week 3 Functions

# Below are defined funcitonf for the three tasks of the applied comp lab week 3, annotated with funcitonal details

# Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import cluster
from sklearn.cluster import KMeans

### Exercise 1

# Defining periodic signal functions

def f1(t, tau1):
    return(np.sin(t-tau1))

def f2(t, tau2):
    return(np.sin(2*t-tau2))

def f3(t, tau3):
    return(np.sin(3*t-tau3))

# f4 is just 1 so doesn't need defined as a separate function

def f_sum(t, tau1, tau2, tau3, a, b, c, d):
    return(a*f1(t,tau1) + b*f2(t,tau2) + c*f3(t,tau3) + d)

# Defining minimise function
def func(x0, args):
    array = args[0]-(x0[0]*args[1]+x0[1]*args[2]+x0[2]*args[3]+x0[3]*args[4])
    output = (array**2).sum()
    return(output)

def signal_minimise(tau_vals, linear_vals, t_final, x0_vals, plot=False):

    # This is a function for finding the defining linear variables of a known complex periodic signal. It takes inputs as follows: tau_vals, a 4 value array of the known tau
    # values of the signal, linear_vals, a 4 value array containing the known weighting of the harmonics of the signal, t_final, an integer value over which the desired
    # timeframe over which the signal is minimised, and x0_vals, a 4 value array containing initial guesses of the linear_vals values. The function then returns the found
    # minimised weighting values, and the shape of the signal determined from this for comparison. There is an additional optional argument to toggle plotting the results.
    
    # Generating input timeframe
    time = np.arange(0,t_final,0.01)

    # Finding function value arrays
    f1_vals = f1(time, tau_vals[0])
    f2_vals = f2(time, tau_vals[1])
    f3_vals = f3(time, tau_vals[2])
    f4_vals = np.ones(len(time))

    # Finding summed values and plotting
    fsum_vals = f_sum(time, *tau_vals, *linear_vals)
    
    # Setting inputs for minimisation
    args_vals = np.stack([fsum_vals, f1_vals, f2_vals, f3_vals, f4_vals])

    # Running mnimise function
    min_vals = minimize(func, x0_vals, args_vals)['x']
    fmin_vals = f_sum(time, *tau_vals, *min_vals)

    if plot==True:
        plt.figure(figsize=(8,5))
        plt.title('Minimised Example Periodic Plot')
        plt.xlabel('Time (s)')
        plt.ylabel('Function Value')
        plt.plot(time,fsum_vals, label='Original Function')
        plt.plot(time, fmin_vals, label='Minimised Value Function')
        plt.grid()
        plt.legend()
        plt.show()
        
    return(min_vals, fmin_vals)

### Task 2

def svd_alg(input_data, load_val):

    # This function applies an svd linear algorithm to a given three dimensional array input_data. It also takes an integer value for load_val, determining the desired how
    # many loadings maps are desired. It then returns the value arrays associated to the data eigenvectors, U, S, and V, and the found loadings up to the desired index.
    
    # Reshaping the data array for use with the linalg function
    shape = input_data.shape
    data_shaped = input_data.reshape((shape[0]*shape[1],shape[2]))

    # Running linalg function
    svd = np.linalg.svd(data_shaped)

    # Defining the variable sets from the function
    U, S, V = svd

    # Finding the loadings of desired index
    loadings_set = []
    for n in range(load_val):
        loadings = (U[:,n]*S[n]).reshape(shape[0],shape[1])
        loadings_set.append(loadings)
    return(U, S, V, loadings_set)

### Task 3

def clustering(input_data, method="KMeans", n_clusters=None, eps=None, min_samples=None):

    # This function takes a two dimensional array inout dataset, and an argument of the clustering method to be applied, either KMeans or DBSCAN, and performs a clustering
    # operation on the dataset as specified. Additionally, depending on the method used, either the number of clusters desired n_clusters, or the cluster spacing eps and 
    # minimum samples to a clister min_samples can be specified for Kmeans and DBSCAN respectively. It returns the coordinates of the cluster sites and the associated labels
    # of each.

    if method=="KMeans":
        # Running clustering algorithm and setting the output variables
        clustering_km = cluster.KMeans(n_clusters=n_clusters).fit(input_data)
        centers_km = clustering_km.cluster_centers_
        labels_km = clustering_km.labels_
        return(centers_km, labels_km)

    elif method == "DBSCAN":
        # Running clustering algorithm
        clustering_db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(input_data)
        labels_db = clustering_db.labels_

        # Remove noise based on labels and determining cluster sites, also formatting to make consistent with KMeans output
        mask = labels_db != -1
        unique_labels = np.unique(labels_db[mask])

        # Returning center values
        centers_db = np.array([
            input_data[labels_db == lbl].mean(axis=0)
            for lbl in unique_labels
        ])

        return centers_db, labels_db

    else:
        raise ValueError("method must be 'KMeans' or 'DBSCAN'")
