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
