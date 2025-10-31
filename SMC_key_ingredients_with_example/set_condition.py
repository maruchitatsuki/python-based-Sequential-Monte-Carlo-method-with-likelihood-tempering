import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
import csv
import pandas as pd
import numba
import time

# Set model information
n_state = 4 # number of all parameters (including sigma)
num_model_params = 4 # number of model parameters
est_params_list = [1, 1, 1, 1] # list indicating which parameters to estimate (1: estimate, 0: fix)
num_est_params = np.sum(est_params_list)
normal_pred = False # If True, prior distribution is normal distribution
n_cores= 30 # number of cores

# Key hyper parameters
random_seed = 1 # Set random seed for reproducibility
n_particle = 1500 # Set number of particles
inv_Np = 1/n_particle
ess_limit = 0.7 # ESS(effective sample size)
r_threshold = 0.7 # threshold for mutation


# Other hyper parameters
mhstep_factor = 0.5
mhstep_factor_cov = 0.5
ad_mhstep_num = 20
mhstep_num = 5
mhstep_ratio = 1.0
r_threshold_f = 0.8
r_threshold_min = 0.1
d_gamma_max = 1
gm_reduction_itr=80
gm_reduction_rate = 0.6
itr_max= 50

# Figure settings
n_hist = 50
fig_dimen = np.int32(n_state*100+11)

# set data
data = pd.read_csv("SMC_key_ingredients_with_example/data_example/obs_y.csv")
n_data = len(data.T)

# set parameter information
sigma_true = 5
baseparams = np.array([13.04, 52.2e3, 1.147e5, 96.7e3, 23.34, -6, 0.72, -2.51e3])
baseparams_withsigma = np.append(baseparams, sigma_true)

use_params = baseparams # 事前分布に用いるパラメータの決定
use_params = np.append(use_params, sigma_true)

high_k = [5,1,3,2,0.1*10,-0.2*10,0.1*10,-0.2*10,2]
low_k  = [2,1,1,1,0.1*10,-0.2*10,0.1*10,-0.2*10, 0.9]

high_limit = use_params + use_params*high_k
low_limit  = use_params - use_params*low_k

