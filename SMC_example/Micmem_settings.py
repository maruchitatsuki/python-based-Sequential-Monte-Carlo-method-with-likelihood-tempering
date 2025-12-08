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

# ==========================================================
# 0. Global Settings
# ==========================================================

# Gaussian prior coefficients (std = coeff * parameter)
coefficent     = np.array([0.5,0.5,0.5])
coefficent_uni = np.array([0.5] * 8)


# ==========================================================
# 2. True Parameters & Prior Range
# ==========================================================
sigma_true = 5

np.random.seed(20250205)

num_est_params = 3                  # model parameters + sigma
num_model_params = 2          # kinetic & transport parameters
est_params_list = [1,1,1]

est_sigma = True

# priors = {
#     "Vmax": {"dist": "normal", "mu": 1.0, "sigma": 0.1},
#     "Km": {"dist": "normal", "mu": 0.0, "sigma": 5.0},
#     "sigma": {"dist": "normal", "mu": 5, "sigma": 10},
# }

# Vmax_true=1.2,
# Km_true=0.5,
priors = {
    "Vmax": {"dist": "uniform", "low": 0, "high": 10},
    "Km": {"dist": "uniform", "low": 0, "high": 10},
    "sigma": {"dist": "uniform", "low": 0, "high": 10},
}

def sample_prior(priors, n_particle):
    result = {}
    for name, p in priors.items():
        if p["dist"] == "normal":
            result[name] = np.random.normal(
                loc=p["mu"], scale=p["sigma"], size=n_particle
            )
        elif p["dist"] == "uniform":
            result[name] = np.random.uniform(
                low=p["low"], high=p["high"], size=n_particle
            )
        else:
            raise ValueError(f"Unknown distribution: {p['dist']}")
    return result

samples = sample_prior(priors, n_particle)
p_pred = np.zeros((n_particle, num_est_params))
for j, name in enumerate(priors.keys()):
    p_pred[:, j] = samples[name] 


itr_max       = 50
n_hist        = 50
fig_dimen     = int(num_est_params * 100 + 11)

w_cov = np.ones((num_est_params, num_est_params))
for i in range(num_est_params):
    w_cov[i, :] = mhstep_factor_cov
    w_cov[i, i] = mhstep_factor


# ==========================================================
# 6. Load Experimental Data
# ==========================================================
dataset = []
n_ex = 6
base_path="data/mm_pseudo_data"
for i in range(0, n_ex):
    df = pd.read_csv(f"{base_path}_{i}.csv")
    data_i = {
        "t": df["t"].values,
        "P_obs": df["P_obs"].values,
        "S0": df["S_true"].iloc[0],
    }
    dataset.append(data_i)
datapoint = len(dataset[0]["t"])
obs_data = dataset

# Empty array to store the posterior (filtered) distribution
p_filt = np.zeros((n_particle, num_est_params))
# At the beginning, all particles have equal weights, each assigned 1/n_particle (the total weight sums to 1)
p_weight = np.ones(n_particle) / n_particle
# Used for resampling
p_is = np.zeros(n_particle, dtype=int)
# Data used in likelihood calculations (e.g., outlet concentrations for each experimental condition);
# shape is number of particles × number of data points
y_cal = np.zeros((n_particle, n_ex))
d_lk = np.zeros(n_particle)
lk1 = np.zeros(n_particle)