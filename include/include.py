import numpy as np
import scipy as sp
from scipy import linalg
import configparser

def parse_inputs(dir,fname):
    # Default input values that are required
    inputs = {'N':10,
              'N_time':10,
              'L':1,
              'TC_depth':0.9,
              'k':1,
              'alpha':1,
              'delta_t':1,
              't_end':20,
              'regularization_method':'Tikhonov'}
    
    raw_inputs = configparser.ConfigParser()
    raw_inputs.optionxform = str
    raw_inputs.read(dir+'/'+fname)   

    for key in raw_inputs['setup']:      
        try:
            inputs[key] = float(raw_inputs['setup'][key])
        except:
            inputs[key] = raw_inputs['setup'][key]

    return inputs

def legendre_init(fname):
    legendre_general = {}

    with open(fname,'r') as f:
        
        i=1

        for line in f:
            header = int(line)
            local_points = np.array([float(j) for j in f.readline().split()])       
            local_weights = np.array([float(j) for j in f.readline().split()])
            
            legendre_general[header] = np.vstack([local_points,local_weights])

            i=i+1

    f.close()
    return legendre_general

def G_gal_global_fun(x,x_0,t,tau,V,gamma,N,L,k,alpha):
    f_vect = f(x,L,N)
    f0_vect = f(x_0,L,N)

    psi = np.matmul(np.transpose(V),f_vect)
    psi0 = np.matmul(np.transpose(V),f0_vect)

    G_gal_global_flat = (k/alpha)*np.matmul(np.transpose(psi*psi0),np.exp(-(gamma[:,np.newaxis])*(np.reshape(t,(1,-1))-np.reshape(tau,(1,-1)))))
    G_gal_global = np.reshape(G_gal_global_flat,(np.size(t,0),-1))

    return G_gal_global

def f(x,L,N):
    N_vect = np.transpose(np.arange(1,N+1,1)[np.newaxis])
    output = (x/L)**N_vect
    return output

def grad_f(x,L,N):
    N_vect = np.transpose(np.arange(1,N+1,1)[np.newaxis])
    output = (N_vect/L)*((x/L)**(N_vect-1))
    return output

def gauss_quadrature(tau_upper,tau_lower,N,legendre_general):
    points = legendre_general[N][0,:]
    weights = legendre_general[N][1,:]

    #print(points)
    tau_diff_mat, points_mat = np.meshgrid(tau_upper-tau_lower,points)
    tau_sum_mat,_ = np.meshgrid(tau_upper+tau_lower,points)

    integrand_mat = (tau_diff_mat)/2*points_mat + (tau_sum_mat)/2*np.ones((np.shape(points_mat)))
    coeff_vect = (tau_upper-tau_lower)/2

    return(coeff_vect,integrand_mat,weights)

def build_GF(N,L,k,alpha,legendre_general):

    L_coeff_vect_eig,L_integrand_mat_eig,L_weights_vect_eig = gauss_quadrature(L,0,4*(N-1),legendre_general)
    f_vect = f(np.transpose(L_integrand_mat_eig),L,N)
    grad_f_vect = grad_f(np.transpose(L_integrand_mat_eig),L,N)

    a = -k*L_coeff_vect_eig*np.matmul((L_weights_vect_eig*grad_f_vect),np.transpose(grad_f_vect))
    b = (k/alpha)*L_coeff_vect_eig*np.matmul((L_weights_vect_eig*f_vect),np.transpose(f_vect))

    a = (a+np.transpose(a))/2
    b = (b+np.transpose(b))/2

    # transposing to stay consistent with matlab conventions. Horrible, I know
    R = np.transpose(np.linalg.cholesky(b))

    if np.size(R) != np.size(b):
        print('Matrix not symmetric positive definite')
        return

    U,D,V_bar = np.linalg.svd(-np.matmul(np.matmul(np.linalg.inv(np.transpose(R)),a),np.linalg.inv(R)))

    #Not sure why but left and right singular vectors are flipped? Might be more consistent w/ Cole and Beck
    V = np.matmul(np.linalg.inv(R),U)
    gamma = D

    return(V,gamma)

def read_q_input(fname,t_vect):
    t_inp = []
    q_inp = []

    with open(fname,'r') as f:
        for line in f:
            lcl_line = line.strip().split(',')

            try:
                t_inp.append(float(lcl_line[0]))
                q_inp.append(float(lcl_line[1]))
            except:
                continue
    
    q = np.transpose(np.interp(t_vect,t_inp,q_inp)[np.newaxis])

    return q