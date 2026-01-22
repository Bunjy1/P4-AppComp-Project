# Defining function for above process

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
