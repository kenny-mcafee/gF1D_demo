from include.include import *
import os
import matplotlib.pyplot as plt

#parse input file
inputs = parse_inputs(os.getcwd(),'input.ini')

#parse through inputs
N = inputs['N']
N_time = inputs['N_time']
L = inputs['L']
TC_depth = inputs['TC_depth']
k = inputs['k']
alpha = inputs['alpha']
delta_t = inputs['delta_t']
t_end = inputs['t_end']

try:
    q_inp_fname = inputs['q_inp_fname']
except:
    q_inp_fname ='q_inp_fname'

# Initial legendre quadrature points and weights
legendre_general = legendre_init('./include/Legendre_roots.dat')

# Build Green's function eigenvectors and eigenvalues
V, gamma = build_GF(N,L,k,alpha,legendre_general)

# building time vectors
t_vect = np.arange(0,t_end+delta_t,delta_t)
t_idx = np.arange(0,len(t_vect))
t_length = len(t_idx)

# building time vectors for numerical convolution (general matrices)
time_coeff_vect = np.zeros((t_length,t_length))
_,time_integrand_mat,time_weights_vect = gauss_quadrature(t_vect[t_idx[1:]],t_vect[t_idx[0:-1]],N_time,legendre_general)

for i in range(1,t_length):
    time_coeff_vect[i,0:i],_,_ = gauss_quadrature(t_vect[t_idx[1:i+1]],t_vect[t_idx[0:i]],N_time,legendre_general)
    
# reshaping time vectors for quick integration (requires Toeplitz structure in Green's function matrix)
time_coeff_vect_quick = time_coeff_vect[1:,0]
t_mat_quick = np.repeat(np.transpose(t_vect[t_idx[1:]][np.newaxis]),len(time_weights_vect),axis=1)
time_integrand_mat_quick = np.repeat(time_integrand_mat[:,0][np.newaxis],t_length-1,axis=0)

# Calculating first column of Green's function matrix
G_discrete_kernel = np.transpose(time_coeff_vect_quick*np.sum(time_weights_vect*G_gal_global_fun(TC_depth,L,t_mat_quick,time_integrand_mat_quick,V,gamma,N,L,k,alpha),axis=1)[np.newaxis])

# Broadcasting first column of Green's function matrix into a Toeplitz matrix structure
zero_pad = np.zeros(t_length-2)
G_discrete = sp.linalg.toeplitz(G_discrete_kernel,np.concatenate((G_discrete_kernel[0],zero_pad)))

# read in input heat flux vector and interpolate to the given time vector 
q = read_q_input(q_inp_fname,t_vect)

# Calculate temperature solution. a couple of notes:
# 1) Coordinate system maters and direction. In the input file, q is negative, as we have defined the surface normal in the positive x-direction
# 2) The Green's function matrix is defined from t = 2:end and tau = 1:end-1. t --> temperature response and tau --> impulse. This is due to the convolution operation
delta_T = -alpha/k*np.matmul(G_discrete,q[0:-1,:])

# Plotting the input heat flux and temperature response
fig, ax = plt.subplots(2,1,figsize=(6, 6))
ax[0].plot(t_vect,-q*10000)
ax[0].set(ylabel='Heat Flux (W/cm^2)')

ax[1].plot(t_vect[1:], delta_T)
ax[1].set(xlabel='Time (s)',ylabel='Temperature (K)')

plt.show()