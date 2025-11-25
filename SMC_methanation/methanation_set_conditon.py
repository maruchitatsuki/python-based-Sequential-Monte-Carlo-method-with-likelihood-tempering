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


est_position = [i for i, x in enumerate(est_params_list) if x == 1]
# print(est_position)
est_position_set = set(est_position)
uni_list_set = set(uni_list)
def_set = uni_list_set - est_position_set


# ==========================================================
# 1. Data Settings 
# ==========================================================
NX = 51
datalist = [0,2,5,6,8,9,10,11,13,14,15,16,17,19,20,21,22,25,
            26,27,28,31,35,38,40,45,49,52,55,58]
# datalist = [0,2]
datastart = datalist[0]
datafin   = datalist[-1]
n_data    = len(datalist)

# ==========================================================
# 2. True Parameters & Prior Range
# ==========================================================
sigma_true = 5
baseparams = np.array([13.04, 52.2e3, 1.147e5, 96.7e3,
                       23.34, -6, 0.72, -2.51e3])

baseparams_withsigma = np.append(baseparams, sigma_true)

use_params = np.append(baseparams, sigma_true)

# upper/lower bound multipliers
high_k = [25,1,30,2,1,-2,1,-2,2]
low_k  = [4,1,4,1,1,-2,1,-2,0.9]

high_limit = use_params + use_params * np.array(high_k)
low_limit  = use_params - use_params * np.array(low_k)
high_limit_array = np.array([high_limit[i] for i in est_position])
low_limit_array = np.array([low_limit[i] for i in est_position])
# ==========================================================
# 3. Physical Constants
# ==========================================================
pi = np.pi
sc = np.array([-4,-1,1,2,0])
Dz = 0.95e-5 #m2/s
rhos = 5075 #kg/m3
Hr = -164940 #J/mol
R = 8.3144589 #J/mol/K
Rr = 0.01/2 #m reactor radius
S = pi*Rr**2 #m2
Cpg = 2800 #J/kg/K 気体の定圧熱容量
Cps = 698 #J/kg/K 触媒
keff = 0.72 #W/(m*K)
dint = 0.005 #m?
U = 68.2480 #W/m2/K
bed = 5.4e-3
ku = 8180
P_stp = 1.013*10**5 #[Pa]

# ==========================================================
# 4. Numerical Tolerances
# ==========================================================
li = []
atol = []
at = 0.001
for i in range(0,7*NX):
    if i < 6*NX:
        li.append(1)
        atol.append(at)
    else:
        li.append(0)
        atol.append(at)
# ==========================================================
# 5. SMC Hyperparameters
# ==========================================================
n_cores        = 30
n_particle     = 1000
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

w_cov = np.ones((num_est_params, num_est_params))
for i in range(num_est_params):
    w_cov[i, :] = mhstep_factor_cov
    w_cov[i, i] = mhstep_factor

# ==========================================================
# 6. Load Experimental Data
# ==========================================================
info_df = pd.read_csv('methanation_data/information.csv').fillna(0)

information = info_df.iloc[datastart:datafin+1].values

Ca_in = np.zeros(n_data)
Cb_in = np.zeros(n_data)
Cc_in = np.zeros(n_data)
Cd_in = np.zeros(n_data)
Ce_in = np.zeros(n_data)
Xa_out = np.zeros(n_data)
Xb_out = np.zeros(n_data)
Xc_out = np.zeros(n_data)
Xd_out = np.zeros(n_data)
Xe_out = np.zeros(n_data)
T_in = np.zeros(n_data)
u_in = np.zeros(n_data)
T_jacket = np.zeros(n_data)
catag = np.zeros(n_data)
reactorlength = np.zeros(n_data)
sccm = np.zeros(n_data)
void = np.zeros(n_data)
Fa_out = np.zeros(n_data)
Fb_out = np.zeros(n_data)
Fc_out = np.zeros(n_data)
Fd_out = np.zeros(n_data)
Fe_out = np.zeros(n_data)

catag = information[:,2]
reactorlength = information[:,4]
T_jacket = information[:,5]
void_fraction = information[:,6]
T_in = information[:,7]
P_total = information[:,9]
in_flow_a = information[:,10]
in_flow_b = information[:,11]
in_flow_c =information[:,12]
in_flow_d =information[:,14]
in_flow_e =information[:,15]
in_flow_total =information[:,16] #sccmのことでもある
out_flow_a = information[:,17]
out_flow_b = information[:,18]
out_flow_c = information[:,19]
out_flow_d = information[:,21]
out_flow_e = information[:,22]
out_flow_total = information[:,23]
out_molf_a = information[:,24]
out_molf_b = information[:,25]
out_molf_c = information[:,26]
out_molf_d = information[:,28]
out_molf_e = information[:,29]

for i in range (0,n_data):
    T_in[i] = T_in[i]+273
    u_in[i] = in_flow_total[i]*1.667e-8/S*(101325*(T_in[i]))/((P_total[i]*1e6+101325)*298)
    Ca_in[i] = (P_total[i]*1e6+101325)/R/T_in[i]*in_flow_a[i]/((in_flow_a[i]+in_flow_b[i]+in_flow_c[i]+in_flow_d[i]+in_flow_e[i]))
    Cb_in[i] = (P_total[i]*1e6+101325)/R/T_in[i]*in_flow_b[i]/((in_flow_a[i]+in_flow_b[i]+in_flow_c[i]+in_flow_d[i]+in_flow_e[i]))
    Cc_in[i] = (P_total[i]*1e6+101325)/R/T_in[i]*in_flow_c[i]/((in_flow_a[i]+in_flow_b[i]+in_flow_c[i]+in_flow_d[i]+in_flow_e[i]))
    Cd_in[i] = (P_total[i]*1e6+101325)/R/T_in[i]*in_flow_d[i]/((in_flow_a[i]+in_flow_b[i]+in_flow_c[i]+in_flow_d[i]+in_flow_e[i]))
    Ce_in[i] = (P_total[i]*1e6+101325)/R/T_in[i]*in_flow_e[i]/((in_flow_a[i]+in_flow_b[i]+in_flow_c[i]+in_flow_d[i]+in_flow_e[i]))
    Xa_out[i] = out_molf_a[i]
    Xb_out[i] = out_molf_b[i]
    Xc_out[i] = out_molf_c[i]
    Xd_out[i] = out_molf_d[i]
    Xe_out[i] = out_molf_e[i]
    # u_in[i] = in_flow_total[i]*1.667e-8/S*(101325+(273+T_in[i]))/(P_total[i]*298)
    T_jacket[i] = T_jacket[i]+273
    catag[i] = catag[i]/1000
    reactorlength[i] = reactorlength[i]/1000
    sccm[i] = in_flow_total[i]
    void[i] = void_fraction[i]
    Fa_out[i] = out_flow_a[i]
    Fb_out[i] = out_flow_b[i]
    Fc_out[i] = out_flow_c[i]
    Fd_out[i] = out_flow_d[i]
    Fe_out[i] = out_flow_e[i]


u_in  = in_flow_total * 1.667e-8 / S * (101325 * T_in) / ((P_total*1e6 + 101325) * 298)

