# Defining function for above process

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
