import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
import csv
import pandas as pd
import numba
import time
np.random.seed(20250205)
num_trueparams = 8
n_state=9
num_model_params =8

est_params_list = [1, 1, 1, 1, 0, 0, 0, 0, 1]
num_est_params = np.sum(est_params_list)
# 基本的にnew_try以外はtrueにしない
taylor = False
normal_pred = False # normal_prd=Trueで正規分布、Falseで一様分布
if est_params_list[-1] == 1:
    est_siguma = True
else:
    est_siguma = False
sigma_true = 3 # sigmaの真値

uni_list = [0,1,2,3,8] # 一様分布にするパラメータ
prior_list_uni = [0,1,2,3,8]

# 事前分布を正規分布にするのであれば分散が必要
# coefficent = np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3])
# coefficent = np.array([0.5,0.5,0.05,0.05,0.5,0.5,0.5,0.5])
coefficent = np.array([0.5,0.5,0.5,0.5,0.3,0.3,0.3,0.3])

# taylorのときにいくつかの分布を一様分布にするときの係数
coefficent_uni = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])


#a:H2 b:CO2 c:CH4 d:H2O e:Ar mole fraction
NX = 51 #分割数
# datalist = [num for num in range(0, 67) if num % 20 == 0]
# datalist = [num for num in range(17, 47)] # 正反応5逆反応5
# datalist = [num for num in range(45, 47)] # 正反応5逆反応5
# datalist = [num for num in range(0, 60)]
# 8-12はデータ不足で除外されており、インデックスの関係上7番までは-1、13番以降は-5
datalist = [4,5,16,17,18,21,23,24,25,26,35,48,51,52,55,57,58] # haltonsampling
datalist = [0, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 26, 27, 28, 31, 35, 38, 40, 45, 49, 52, 55, 58] # クラスタリング

# datalist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
datastart = datalist[0] # ここから
datafin = datalist[-1] # ここまで
# datapoint = len(datalist)
n_data = len(datalist)
print(n_data)
# 正反応は31逆反応33検証用？は2-2の順で並んでいる。

n_cores= 30 # コア数
n_particle = 1000 # 粒子数

# params_MAP_68 = np.array([ 8.40719518e+00  ,4.87058304e+04 , 1.86905485e+05  ,7.74128612e+04 ,1.89333738e+01 ,-1.76609118e+01  ,3.63368329e-01  ,1.78716572e+04 ,9.61347705e+00])
# params_MAP_1 = np.array([1.13314865e+01 ,5.43277698e+04 ,1.85008121e+05 ,8.95663192e+04, 3.13464709e+01 ,6.24067973e+00 ,8.89829545e-02 ,2.44583252e+03,8.94418914e+00])
trueparams = np.array([13.04, 52.2e3, 1.147e5, 96.7e3, 23.34, -6, 0.72, -2.51e3])
# trueparams = np.array([6.196262,49473.01,236128.1,101431.7,22.20363,-7.58698,0.661501,1036.961])
trueparams_withsigma = np.append(trueparams, sigma_true)
# trueparams = np.array([14.1, 46.98e3, 1.6e5, 92e3, 35, 1500, 1.8, 2.61e3])



use_params = trueparams # 事前分布に用いるパラメータの決定
use_params = np.append(use_params, sigma_true)
high_k = [25,1,30,2,0.1*10,-0.2*10,0.1*10,-0.2*10,2]
low_k  = [4,1,4,1,0.1*10,-0.2*10,0.1*10,-0.2*10,0.9]
# high_k = [10,1,12,2,0.1*10,-0.2*10,0.1*10,-0.2*10,2]
# low_k  = [4,1,4,1,0.1*10,-0.2*10,0.1*10,-0.2*10,0.9]
# high_k = [5,1,3,2,0.1*10,-0.2*10,0.1*10,-0.2*10,2]
# low_k  = [2,1,1,1,0.1*10,-0.2*10,0.1*10,-0.2*10, 0.9]
# high_k = [1*10,1*10,2*10,1*10,0.1*10,-0.2*10,0.1*10,-0.2*10,2]
# low_k  = [1*10,1*10,2*10,1*10,0.1*10,-0.2*10,0.1*10,-0.2*10,0.09]
high_limit = use_params + use_params*high_k
low_limit  = use_params - use_params*low_k
# print(high_limit,low_limit)
# high_limit =  np.array([13.04, 52.2e3, 1.147e5, 96.7e3, 23.34, -6, 0.72, -2.51e3,0.5]) + np.array([13.04, 52.2e3, 8*1.147e4, 96.7e2, 23.34, 300*10, 3*0.72, 10*2.51e3,0.01])*0.5
# low_limit =   np.array([13.04, 52.2e3, 1.147e5, 96.7e3, 23.34, -6, 0.72, -2.51e3,0]) - np.array([13.04, 52.2e3, 8*1.147e4, 96.7e2, 23.34, 300*10, 3*0.72, 10*2.51e3,0])*0.5

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

difference_exp = 1

n_hist = 50
inv_Np = 1/n_particle
itr_max= 50
fig_dimen = np.int32(n_state*100+11)
mhstep_factor = 0.5
mhstep_factor_cov = 0.5
ad_mhstep_num = 20
mhstep_num = 5
mhstep_ratio = 1.0
r_threshold = 0.5
r_threshold_f = 0.7
r_threshold_min = 0.1
d_gamma_max = 1
gm_reduction_itr=80
gm_reduction_rate = 0.7
ess_limit = 0.5

## print(Pa_in)
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
# print(len(T_in))



# CSVファイルを読み込み、DataFrameとして取得
# 8~11まで抜けていると思っていたが、T_inがunmeasurementのため意図的に抜いていたようだ
information_df = pd.read_csv('gpromscsv/information.csv').fillna(0)
concentration_gproms_df = pd.read_csv('gpromscsv/concentration_all_gproms.csv').fillna(0)
information = information_df.iloc[datastart:datafin+1].values
# concentration_gproms = concentration_gproms_df.iloc[datastart:datafin+1].values


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

for i in range (n_data):
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

    # C_all[i,0*NX_g:1*NX_g] = Ca_gproms[i]
    # C_all[i,1*NX_g:2*NX_g] = Cb_gproms[i]
    # C_all[i,2*NX_g:3*NX_g] = Cc_gproms[i]
    # C_all[i,3*NX_g:4*NX_g] = Cd_gproms[i]
    # C_all[i,4*NX_g:5*NX_g] = Ce_gproms[i]