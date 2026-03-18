""" Compiled set of functions developed during the first three weeks of lab sessions.

    Each function includes a description along with required inputs and outputs.
    Examples of usage can be found in the related example notebook.
"""

""" Importing Required Dependencies."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn import cluster
from sklearn.cluster import KMeans

""" Week 1 Functions """
def double(x):
    """ Simple testbed function for becoming familiar with Git.
        Return the input value multiplied by two.

        Parameters
        ----------
        x: float or int
            The number to be doubled.
    
        Returns
        -------
        y: float or int
            The input value multiplied by 2.
        """
    y = 2*x
    return(y)

""" Week 2 Functions """
""" The periodic_fitter function developed this week has been omitted here and moved to the project work functions file as it is relevant there. """

def straight_line(x, m, c):
    """Function for plotting a straight line, primarily for use in curve fitting.

        Parameters
        ----------
        x : array-like
            The x values of the dataset for plotting.
        m : float
            The gradient (slope) of the straight line.
        c : float
            The y-axis intercept of the straight line.
    
        Returns
        -------
        y : array-like
            The calculated y values of the straight line corresponding to x.
        """
    y = m * x + c
    return(y)

def straightline_fitter(x_data, y_data, plotting=False):
    """Fit a straight line to a 2D dataset and optionally plot the results.

        This function fits a straight line to the supplied x and y data using
        scipy's curve_fit. It calculates the gradient and intercept of the
        best-fitting line, with associated uncertainties, residuals, and R² value.
        Optionally, the input data and found fitted line can be plotted.
    
        Parameters
        ----------
        x_data : numpy.ndarray
            The x-values of the dataset.
        y_data : numpy.ndarray
            The y-values of the dataset.
        plotting : bool, optional
            If True, the function generates plots of the fitted line and the
            residuals. Default is False.
    
        Returns
        -------
        params : numpy.ndarray
            The fitted parameters of the straight line [gradient, intercept].
        errors : numpy.ndarray
            The uncertainties of the fitted parameters [gradient_error, intercept_error].
        r_squared : float
            The coefficient of determination (R²) indicating the goodness of fit.
        """

    # Error checking for matching array lengths
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")

    # Run optimize curve fit on data and define results
    params, params_covariance = optimize.curve_fit(straight_line, x_data, y_data, p0=[1, 0])

    grad, y_int = params
    errors = np.sqrt(np.diag(params_covariance))
    grad_err, y_int_err = errors

    # Calculating the fitted y-values to find the residuals and R²
    y_fit = straight_line(x_data, grad, y_int)
    residuals = y_fit - y_data

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - ss_res / ss_tot

    # Optional plotting for results
    if plotting:
        plt.figure(figsize=(10, 8))
        plt.suptitle("Raw Data and Fitted Line Plot, with Residuals and Parameters")

        plt.subplot(2, 1, 1)
        plt.scatter(x_data, y_data, marker="x", color="black", label="Raw Data")
        plt.plot(x_data, y_fit, color="red", label='Best Fitting Line \nParameters: \nm={:.5g}\u00B1{:.2g} \nc={:.3g}\u00B1{:.2g} \nR\u00B2={:.4g}'.format(grad, grad_err, y_int, y_int_err, r_squared))
        plt.title("Raw Data and Fitted Line")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(x_data, residuals, label="Residuals")
        plt.axhline(0, linestyle="--")
        plt.title("Residuals Plot")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    return(params, errors, r_squared)

"""Defining a set of polynomial functions of order 1 through 6.
   These are primarily for use in the polynomial_fitter function.
   This could be optimised in future by implementing a generalised polynomial function.
   """
def order1(x, a, b):
    """ First order polynomial. """
    return(a + b*x)

def order2(x, a, b, c):
    """ Second order polynomial. """
    return(a + b*x + c*x**2)

def order3(x, a, b, c, d):
    """ Third order polynomial. """
    return(a + b*x + c*x**2 + d*x**3)

def order4(x, a, b, c, d, e):
    """ Fourth order polynomial. """
    return(a + b*x + c*x**2 + d*x**3 + e*x**4)

def order5(x, a, b, c, d, e, f):
    """ Fifth order polynomial. """
    return(a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5)

def order6(x, a, b, c, d, e, f, g):
    """ Sixth order polynomial. """
    return(a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6)
    
def polynomial_fitter(x_data, y_data, order, params_guess, plotting=False):
    """Fit a polynomial curve of chosen order between 1 and 6 to a 2D dataset and optionally plot the results.

        This function fits a polynomial to the supplied x and y data using
        scipy's curve_fit. It calculates the best-fitting polynomial 
        parameters, with associated uncertainties, residuals, and R² value.
        Optionally, the input data and found fitted line can be plotted.
    
        Parameters
        ----------
        x_data : numpy.ndarray
            The x-values of the dataset.
        y_data : numpy.ndarray
            The y-values of the dataset.
        order : int
            The order of the polynomial to be fitted. Must be between 1 and 6.
        params_guess : numpy.ndarray or list
            Initial guesses for the polynomial parameters used by curve_fit.
            The length must match the number of parameters required for the
            chosen polynomial order.
        plotting : bool, optional
            If True, the function generates plots of the fitted polynomial and
            the residuals. Default is False.
    
        Returns
        -------
        params : numpy.ndarray
            The fitted parameters of the polynomial.
        errors : numpy.ndarray
            The uncertainties of the fitted parameters calculated from the
            covariance matrix.
        r_squared : float
            The coefficient of determination (R²) indicating the goodness of fit.
        """

    # Error if statements to allow 'failing with grace'
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")

    if order < 1 or order > 6:
        raise ValueError("order must be an integer between 1 and 6")

    if len(params_guess) != order + 1:
        raise ValueError("params_guess must contain order + 1 parameters")

    # Defining polynomial order options and selecting one from the input
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

    # Optional curve plotting
    if plotting:
        # Setting up subplots
        plt.figure(figsize=(10,8))
        plt.suptitle('Raw Data and Fitted Line Plot, with Residuals and Line Fit Parameters')
        
        # Designing graph
        plt.subplot(2,1,1)
        plt.scatter(x_data, y_data, marker='x', color='black', label='Raw Data')
        plt.title('Raw Data and Fitted Line Plot')
        
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

    return(params, errors, r_squared)


""" Week 3 Functions """

"""Defining a set of signal functions, their sum, and a general minimising function.
   These are primarily for use in the signal_minimise function and have
   limited use outside this context.
"""
    
def f1(t, tau1):
    return(np.sin(t - tau1))

def f2(t, tau2):
    return(np.sin(2*t - tau2))

def f3(t, tau3):
    return(np.sin(3*t - tau3))

# f4 is just 1 so doesn't need defined as a separate function

def f_sum(t, tau1, tau2, tau3, a, b, c, d):
    return(a*f1(t, tau1) + b*f2(t, tau2) + c*f3(t, tau3) + d)

def func(x0, args):
    """ Objective function used for signal minimisation.

        Computes the sum of squared residuals between the observed signal
        and a combintation of the four tau functions above.
    
        Parameters
        ----------
        x0 : array-like
            Coefficients applied to the basis functions.
        args : tuple
            Contains the observed signal and the basis function arrays.
    
        Returns
        -------
        cost : float
            Sum of squared residuals to be minimised.
        """

    y, f1_vals, f2_vals, f3_vals, f4_vals = args

    array = y - (x0[0]*f1_vals + x0[1]*f2_vals + x0[2]*f3_vals + x0[3]*f4_vals)
    square_sum = (array**2).sum()
    return(square_sum)

def signal_minimise(tau_vals, linear_vals, t_final, x0_vals, plot=False):
    """ This function reconstructs a known periodic signal by use of minimisation.
        There is not much practical utility here as to reconstruct the signal the
        original parameters of its generation must be known, but it serves as an
        introduction to the process for the project.
    
        Parameters
        ----------
        tau_vals : array-like
            Array containing the tau values of each function component.
        linear_vals : array-like
            The original weighting values of each 
        t_final : int or float
            End time value of the signal on the x-axis.
        x0_vals : array-like
            Initial guesses of the linear_vals array for minimisation.
        plot : bool, optional
            Toggle to plot the results of the function.
    
        Returns
        -------
        min_vals : numpy.ndarray
            The minimised weighting values found by the function.
        fmin_vals : numpy.ndarray
            The reconstructed signal using min_vals.
        """

    # Error messages for wrong inputs
    if len(tau_vals) != 3:
        raise ValueError("tau_vals must contain 3 values for the signal phase offsets")

    if len(linear_vals) != 4:
        raise ValueError("linear_vals must contain 4 values for the harmonic weights")

    if len(x0_vals) != 4:
        raise ValueError("x0_vals must contain 4 initial guesses for the minimisation")

    if t_final <= 0:
        raise ValueError("t_final must be a positive value")

    if not isinstance(plot, bool):
        raise TypeError("plot must be True or False")

    # Generating input timeframe
    time = np.arange(0, t_final, 0.01)

    # Finding function value arrays
    f1_vals = f1(time, tau_vals[0])
    f2_vals = f2(time, tau_vals[1])
    f3_vals = f3(time, tau_vals[2])
    f4_vals = np.ones(len(time))

    # Finding summed values and plotting
    fsum_vals = f_sum(time, *tau_vals, *linear_vals)
    
    # Setting inputs for minimisation
    args_vals = np.stack([fsum_vals, f1_vals, f2_vals, f3_vals, f4_vals])

    # Running minimise function
    min_vals = optimize.minimize(func, x0_vals, args=(args_vals,))['x']
    fmin_vals = f_sum(time, *tau_vals, *min_vals)

    if plot == True:
        plt.figure(figsize=(8,5))
        plt.title('Minimised Example Periodic Plot')
        plt.xlabel('Time (s)')
        plt.ylabel('Function Value')
        plt.plot(time, fsum_vals, label='Original Function')
        plt.plot(time, fmin_vals, label='Minimised Value Function')
        plt.grid()
        plt.legend()
        plt.show()
        
    return(min_vals, fmin_vals)
    
def svd_alg(input_data, load_val):
    """ This function applies Numpy's singular value decomposition (SVD) algorithm to a 
        3D dataset. It computes the SVD and returns the matrices U, S, and V, with a set
        of loading maps derived from the specified number of components.
    
        Parameters
        ----------
        input_data : numpy.ndarray
            A three-dimensional array containing the input dataset.
        load_val : int
            The desired number of loading maps.
    
        Returns
        -------
        U : numpy.ndarray
            Left singular vectors of the reshaped data matrix.
        S : numpy.ndarray
            Singular values corresponding to the decomposition.
        V : numpy.ndarray
            Right singular vectors of the reshaped data matrix.
        loadings_set : list
            A list containing the calculated loading maps up to the specified index.
        """

    # Error messages for inputs
    if not isinstance(input_data, np.ndarray):
        raise TypeError("input_data must be a numpy.ndarray")

    if input_data.ndim != 3:
        raise ValueError("input_data must be a three-dimensional array")

    if not isinstance(load_val, int):
        raise TypeError("load_val must be an integer")

    if load_val <= 0:
        raise ValueError("load_val must be a positive integer")

    if load_val > input_data.shape[2]:
        raise ValueError("load_val cannot exceed the number of available components")

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
    """ This function applies a clustering algorithm to a two-dimensional dataset.
        The clustering method used can be either KMeans or DBSCAN, with appropriate
        parameters needed for either as required below. The function utilises the
        inbuilt clustering algorithms of the sklearn package.
    
        Parameters
        ----------
        input_data : numpy.ndarray
            A two-dimensional array containing the dataset to be clustered.
        method : str, optional
            The clustering algorithm to use. Must be either "KMeans" or "DBSCAN".
            Default is "KMeans".
        n_clusters : int, optional
            The number of clusters to form (KMeans specific).
        eps : float, optional
            The maximum distance between samples for them to be considered part of
            the same neighbourhood (DBSCAN specific).
        min_samples : int, optional
            The minimum number of samples required to form a cluster (DBSCAN specific).
    
        Returns
        -------
        centers : numpy.ndarray
            The coordinates of the cluster centres.
        labels : numpy.ndarray
            The cluster label assigned to each input data point.
        """

    # Generic error handling related to input data
    if not isinstance(input_data, np.ndarray):
        raise TypeError("input_data must be a numpy.ndarray")

    if input_data.ndim != 2:
        raise ValueError("input_data must be a two-dimensional array")

    if method not in ["KMeans", "DBSCAN"]:
        raise ValueError("method must be 'KMeans' or 'DBSCAN'")

    # Running algorithm with KMeans method
    if method == "KMeans":
        # Additional error flags specific to KMeans
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans")

        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")

        # Running clustering algorithm and setting the output variables
        clustering_km = cluster.KMeans(n_clusters=n_clusters).fit(input_data)
        centers_km = clustering_km.cluster_centers_
        labels_km = clustering_km.labels_

        return(centers_km, labels_km)

    # Running algorithm with DBSCAN method
    elif method == "DBSCAN":
        # Additional error flags specific to DBSCAN
        if eps is None or min_samples is None:
            raise ValueError("eps and min_samples must be specified for DBSCAN")

        if eps <= 0:
            raise ValueError("eps must be a positive value")

        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")

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
        # Returning an error if neither specified algorithm is used
        raise ValueError("method must be 'KMeans' or 'DBSCAN'")
