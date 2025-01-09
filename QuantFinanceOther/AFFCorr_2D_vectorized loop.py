import numpy as np
import scipy.linalg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import time
# t_start=time.time()


# The code snippet you provided is setting up the correlation matrix, average values (avgs), standard
# deviations (stdevs), and weights for a financial simulation or analysis. Here's a breakdown of what
# each of these variables represents:
correlation_matrix = np.array([
    [1, 0.690395261, 0.706442335, 0.5, 0.325072398],
    [0.690395261, 1, 0.769745281, 0.5, 0.313234577],
    [0.706442335, 0.769745281, 1, 0.5, 0.063954621],
    [0.5, 0.5, 0.5, 1, 0],
    [0.325072398, 0.313234577, 0.063954621, 0, 1]
])

avgs=np.array([0.0,0.0,0.0,0.0,0.0])
stdevs=np.array([0.0778, 0.0654, 0.0634, 0.0302, 0.1228])

weights=np.array([0.62, -0.98, 0.2, 0.12, -0.02])
# w poniższym trzeba pamiętać żeby pozycje kosztowe wprowadzać z '-' a przychodowe z '+'
# kolejność aktywów jest ściśle określona i nie należy jej zmieniać, 
# bo jest powiązana z stdev, weights i macierzą korelacji
assets_values =  np.array([
    [101553812, -127258733, 38656557, 12039749, -2359694],
    [96447068, -123080826, 36170888, 12017941, -2308763],
    [147151631, -186767998, 54724693, 18065610, -3463144]
])
# defines hedge levels
# in the future can be developed as multi-dimensional array, with separate hedge ratios for each product
hedge_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])



n_simul=10000
# simul_1M=np.zeros((n_simul,11))
# simul_2M=np.zeros((n_simul,11))
# simul_3M=np.zeros((n_simul,11))
# adjusted_random_matrix=np.zeros((5,3))

#na potrzeby testów robię jeden zestaw liczb losowych, do usunięcia potem
# random_normal_matrix = np.array([ [2.195082178, 0.321369056, 0.830013604], 
#                   [0.071134897, 1.545776666, -0.242618127], 
#                   [-1.023405683, -1.381506951, -3.295318299], 
#                   [-0.120304182, 0.788909078, 0.360224694], 
#                   [-0.650127992, -1.399728369, 0.422675004] ])

# Perform Cholesky decomposition to get the lower triangular matrix
L_matrix = np.linalg.cholesky(correlation_matrix)

i=0

# Generate uncorrelated random normal variables (standard normal distribution) for all simulations at once
random_normal_matrix = np.random.normal(size=(n_simul, L_matrix.shape[0], 3))

# Adjust random variables with mean and standard deviation
adjusted_random_matrix = np.zeros_like(random_normal_matrix)
adjusted_random_matrix[:,:,0] = avgs + stdevs * random_normal_matrix[:,:,0]
adjusted_random_matrix[:,:,1] = adjusted_random_matrix[:,:,0] + avgs + stdevs * random_normal_matrix[:,:,1]
adjusted_random_matrix[:,:,2] = adjusted_random_matrix[:,:,1] + avgs + stdevs * random_normal_matrix[:,:,2]

# Multiply the L_matrix by the adjusted_random_matrix to obtain correlated random variables
correlated_random_matrix = np.einsum('ij, sik -> sjk', L_matrix, adjusted_random_matrix)

# # Compute the exponential of hedge levels
exp_hedge_levels_1M = np.exp(np.outer((1-hedge_levels), np.ones(n_simul))[:, :, np.newaxis] * correlated_random_matrix[:,:,0])
# exp_hedge_levels_2M = np.exp(np.outer((1-hedge_levels), np.ones(n_simul))[:, :, np.newaxis] * correlated_random_matrix[:,:,1])
# exp_hedge_levels_3M = np.exp(np.outer((1-hedge_levels), np.ones(n_simul))[:, :, np.newaxis] * correlated_random_matrix[:,:,2])
# exp_hedge_levels_1M = np.exp(np.outer((1-hedge_levels),correlated_random_matrix[:,0]))


# # Broadcast multiplication across the arrays
# hedged_new_asset_values_1M = exp_hedge_levels_1M * assets_values[0,:]
# hedged_new_asset_values_2M = exp_hedge_levels_2M * assets_values[1,:]
# hedged_new_asset_values_3M = exp_hedge_levels_3M * assets_values[2,:]

# # Calculate the margins vector as the sum across each row
# margins_vector_1M = hedged_new_asset_values_1M.sum(axis=2)
# margins_vector_2M = hedged_new_asset_values_2M.sum(axis=2)
# margins_vector_3M = hedged_new_asset_values_3M.sum(axis=2)

# # Save the result of each simulation in a separate table
# simul_1M[:,:] = margins_vector_1M.T
# simul_2M[:,:] = margins_vector_2M.T
# simul_3M[:,:] = margins_vector_3M.T

x=pd.DataFrame(correlated_random_matrix[:,:,0])
y=pd.DataFrame(exp_hedge_levels_1M[:,:,0])
x.to_excel('x.xlsx', index=False)
y.to_excel('y.xlsx', index=False)