"""
=========================================================
 Configuration File for SMC-based Parameter Estimation
 System: Methanation Reactor Model
 Author: (your name)
=========================================================
"""

import numpy as np
import pandas as pd

# ==========================================================
# 0. Global Settings
# ==========================================================
np.random.seed(20250205)

n_state = 9                   # model parameters + sigma
num_model_params = 8          # kinetic & transport parameters
est_params_list = [1,1,1,1,0,0,0,0,1]
num_est_params = np.sum(est_params_list)

taylor       = False
normal_pred  = False          # False: uniform, True: Gaussian prior
est_sigma    = est_params_list[-1] == 1

# Parameters using uniform prior
uni_list = [0,1,2,3,8]

# Gaussian prior coefficients (std = coeff * parameter)
coefficent     = np.array([0.5,0.5,0.5,0.5,0.3,0.3,0.3,0.3])
coefficent_uni = np.array([0.5] * 8)

# ==========================================================
# 1. Data Settings 
# ==========================================================
NX = 51
datalist = [0,2,5,6,8,9,10,11,13,14,15,16,17,19,20,21,22,25,
            26,27,28,31,35,38,40,45,49,52,55,58]
datastart = datalist[0]
datafin   = datalist[-1]
n_data    = len(datalist)

print(f"Number of data points used: {n_data}")

# ==========================================================
# 2. True Parameters & Prior Range
# ==========================================================
sigma_true = 3
trueparams = np.array([13.04, 52.2e3, 1.147e5, 96.7e3,
                       23.34, -6, 0.72, -2.51e3])

trueparams_withsigma = np.append(trueparams, sigma_true)

use_params = np.append(trueparams, sigma_true)

# upper/lower bound multipliers
high_k = [25,1,30,2,1,-2,1,-2,2]
low_k  = [4,1,4,1,1,-2,1,-2,0.9]

high_limit = use_params + use_params * np.array(high_k)
low_limit  = use_params - use_params * np.array(low_k)

# ==========================================================
# 3. Physical Constants
# ==========================================================
pi = np.pi
Dz = 0.95e-5
rhos = 5075
Hr = -164940
R = 8.3144589
Rr = 0.01 / 2
S = pi * Rr**2
Cpg = 2800
Cps = 698
keff = 0.72
U = 68.2480
bed = 5.4e-3
ku = 8180
P_stp = 1.013e5

# ==========================================================
# 4. Numerical Tolerances
# ==========================================================
atol = []
for i in range(0, 7 * NX):
    atol.append(0.001)

# ==========================================================
# 5. SMC Hyperparameters
# ==========================================================
n_cores        = 30
n_particle     = 30
inv_Np         = 1 / n_particle
ess_limit      = 0.5

mhstep_factor      = 0.5
mhstep_factor_cov  = 0.5
ad_mhstep_num      = 20
mhstep_num         = 5
mhstep_ratio       = 1.0

r_threshold        = 0.5
r_threshold_f      = 0.7
r_threshold_min    = 0.1
d_gamma_max        = 1
gm_reduction_itr   = 80
gm_reduction_rate  = 0.7

itr_max       = 50
n_hist        = 50
fig_dimen     = int(n_state * 100 + 11)

# ==========================================================
# 6. Load Experimental Data
# ==========================================================
info_df = pd.read_csv('methanation_data/information.csv').fillna(0)

information = info_df.iloc[datastart:datafin+1].values

# Pre-allocate arrays
T_in       = information[:,7] + 273
T_jacket   = information[:,5] + 273
catag      = information[:,2] / 1000
reactorlen = information[:,4] / 1000
void       = information[:,6]
P_total    = information[:,9]

inA, inB, inC = information[:,10], information[:,11], information[:,12]
inD, inE      = information[:,14], information[:,15]
in_total      = information[:,16]

outA, outB, outC = information[:,17], information[:,18], information[:,19]
outD, outE       = information[:,21], information[:,22]
out_total        = information[:,23]

Xa = information[:,24]
Xb = information[:,25]
Xc = information[:,26]
Xd = information[:,28]
Xe = information[:,29]

# Convert flows → concentrations
Ca_in = (P_total*1e6+101325)/R/T_in * inA/(inA+inB+inC+inD+inE)
Cb_in = (P_total*1e6+101325)/R/T_in * inB/(inA+inB+inC+inD+inE)
Cc_in = (P_total*1e6+101325)/R/T_in * inC/(inA+inB+inC+inD+inE)
Cd_in = (P_total*1e6+101325)/R/T_in * inD/(inA+inB+inC+inD+inE)
Ce_in = (P_total*1e6+101325)/R/T_in * inE/(inA+inB+inC+inD+inE)

u_in  = in_total * 1.667e-8 / S * (101325 * T_in) / ((P_total*1e6 + 101325) * 298)

print("Configuration loaded successfully.")
