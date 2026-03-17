""" Complete function to apply processes designed in tfp and decomp files.
    Represents summative work of the project.
    
    Each function includes a description along with required inputs and outputs.
    Examples of usage can be found in the related example notebook.
"""

""" Importing Required Dependencies."""
import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

""" Function """
def fit_compare(dataset, vectors, params_guess, bounds, algorithm='SVD', metrics=False, component_plots=False, param_plots=False, residuals=False, iterative_fitting=True, clean_outliers=False, save_data=False, filename='placeholder_name'):

    # Finding vector decomps
    raw_data = dataset.data
    clean_hsdata = auto_decomp(dataset, vectors, algorithm=algorithm, metrics=metrics)
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
        raw_norm = normalize_intensity(raw_avg)
        
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
    raw_fit = perfit_iterate(raw_data, params_guess, bounds, iterative_fitting=iterative_fitting)
    clean_fit = perfit_iterate(clean_data, params_guess, bounds, iterative_fitting=iterative_fitting)

    # Setting paramaters
    raw_params = raw_fit[0]
    raw_errors = raw_fit[1]
    raw_r2 = raw_fit[2]
    #print(raw_errors.shape)

    clean_params = clean_fit[0]
    clean_errors = clean_fit[1]
    clean_r2 = clean_fit[2]

    # Adjusting raw errors to exclude overly large values
    if clean_outliers == True:
        outlier_cleaning(raw_errors)
        outlier_cleaning(clean_errors)

    # Params plots
    if param_plots == True:
        param_plotting(raw_params, clean_params, 'Parameter', residuals=residuals)
        param_plotting(raw_errors, clean_errors, 'Error', residuals=residuals)

    # Saving data into a .npy file
    if save_data == True:
        full_data = np.array([raw_params, raw_errors, clean_params, clean_errors])
        full_data = np.stack(full_data)
        np.save(filename, full_data)

    return(raw_data, raw_params, raw_errors, raw_r2, clean_data, clean_params, clean_errors, clean_r2)
