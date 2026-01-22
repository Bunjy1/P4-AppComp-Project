# Creating a periodic fitter function
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
            ss_tot = sum((testset1-np.mean(testset1))**2)
            r_squared = 1 - ss_res/ss_tot

            params_array.append(params)
            r2_array.append(r_squared)

            #plt.plot(params)
            #plt.scatter(run, params[0])
            #print('Run {}'.format(run))
            run = run+1
    params_array = np.stack(params_array)
    return(params_array, r2_array)
