import numpy as np
import scipy.linalg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D

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
assets_values = np.array([150842716, -186953956, 57389058, 17686620, -3675825])
# defines hedge levels
# in the future can be developed as multi-dimensional array, with separate hedge ratios for each product
hedge_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])



n_simul=10000
simul=np.zeros((n_simul,11))

i=0

for i in range(0,n_simul):
    # Perform Cholesky decomposition to get the lower triangular matrix
    L_matrix = np.linalg.cholesky(correlation_matrix)
    # Generate uncorrelated random normal variables (standard normal distribution)
    random_normal_vector = np.random.normal(size=L_matrix.shape[0])
    # Apply the mean and standard deviation to the random variables
    adjusted_random_vector = avgs + stdevs * random_normal_vector
    # Multiply the L_matrix by the adjusted_random_vector to obtain correlated random variables
    correlated_random_vector = np.dot(L_matrix, adjusted_random_vector)
    # Compute the exponential of hedge levels
    exp_hedge_levels = np.exp(np.outer((1-hedge_levels),correlated_random_vector))
    # Broadcast multiplication across the arrays
    hedged_new_asset_values = exp_hedge_levels*assets_values
    # Calculate the margins vector as the sum across each row
    margins_vector = hedged_new_asset_values.sum(axis=1)
    # calculates the margin in USD
    simul[i,:]=margins_vector
    

a=pd.DataFrame(correlated_random_vector)
b=pd.DataFrame(exp_hedge_levels)
c=pd.DataFrame(hedged_new_asset_values)
d=pd.DataFrame(margins_vector)
e=pd.DataFrame(simul, columns=[f'Hedge_{int(h*100)}%' for h in hedge_levels])


# saves the results of the simulation into Excel
a.to_excel('a.xlsx', index=False)
b.to_excel('b.xlsx', index=False)
c.to_excel('c.xlsx', index=False)
d.to_excel('d.xlsx',index=False)
e.to_excel('e.xlsx',index=False)
s
# plots results
# plt.hist(simul[:,0], bins=100, edgecolor='black')
# plt.title('Margin-at-risk')
# plt.xlabel('Margin in USD')
# plt.ylabel('Frequency')
# plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
# plt.show()

