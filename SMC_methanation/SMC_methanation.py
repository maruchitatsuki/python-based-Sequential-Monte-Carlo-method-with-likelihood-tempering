# import modules
# === Standard Library ===
import os
import sys
import time
import datetime
import json
import traceback
import psutil
import requests

# === Numerical & Scientific Computation ===
import numpy as np
import scipy
from scipy import integrate, optimize

# === Visualization ===
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as P

# === Data Handling ===
import pandas as pd

# === Parallelization & Optimization ===
from joblib import Parallel, delayed
import numba
import ray

# === Assimulo (DAE Solvers) ===
from assimulo.solvers import Radau5DAE, IDA
from assimulo.problem import Implicit_Problem

# === Custom Modules ===
from SMC_methanation_data import *

# === Profiling ===
from memory_profiler import profile



# 各種保存用のディレクトリの作成
dirname = f"methanation_save/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{n_data}/" #Name of the folder (directory) you want to create
os.makedirs(dirname, exist_ok=True)  # Create a folder

dirnameProgress = f"{dirname}tubular_Histgram_Progress/"
os.makedirs(dirnameProgress, exist_ok=True)

dirnametubular = f"{dirname}tubular_Progress/"
os.makedirs(dirnametubular, exist_ok=True)

dirnametubular_mean = f"{dirname}tubular_Progress_means/"
os.makedirs(dirnametubular_mean, exist_ok=True)

dirnamepred = f"{dirname}pred/"
os.makedirs(dirnamepred, exist_ok=True)

dirnameC_l_ = f"{dirname}C_l_/"
os.makedirs(dirnameC_l_, exist_ok=True)

dirnamevaliables = f"{dirname}valiables/"
os.makedirs(dirnamevaliables,exist_ok=True)

# SMC_tsuboi_dataを読み込み，変数に格納，別の変数としてtxtファイルを定義して書き込みを行う
with open("SMC_methanation_data.py", mode="r", encoding="utf-8_sig") as THISFILE:
    CODE = THISFILE.read()
with open(f"{dirname}Initdata_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mode="w") as PROGRAM:
    PROGRAM.write(CODE)

# 共分散行列を定義？
w_cov = np.ones((num_est_params, num_est_params))
for i in range(num_est_params):
    w_cov[i, :] = mhstep_factor_cov
    w_cov[i, i] = mhstep_factor


# 反応速度式の計算
@numba.jit("f8(f8,f8,f8,f8,f8,f8[:])",nopython=True)
def func_rCH4(T,Ca,Cb,Cc,Cd,params):
    PH2 = Ca*R*T*10**(-6)
    PCO2 = Cb*R*T*10**(-6)
    PCH4 = Cc*R*T*10**(-6)
    PH2O = Cd*R*T*10**(-6)
    kf = params[0]*np.exp(-params[1]/R/T)
    ks = params[2]*np.exp(-params[3]/R/T)
    kCO2 = params[4]*np.exp(-params[5]/R/T)
    kH2O = params[6]*np.exp(-params[7]/R/T)
    rf = 5075e3*kf*kCO2*PCO2*(max(0.001,PH2)**0.5)/((1+kCO2*PCO2)**2)
    rr = 5075e3*ks*kH2O*PH2O*(PCH4**2)/((1+kH2O*PH2O)**2)
    rCH4 = (rf-rr)

    return rCH4

# 気体密度の計算
@numba.jit("f8(f8,f8,f8,f8,f8,f8,f8)",nopython=True)
def func_rohg(a,b,c,d,e,T,P0):
    rohg = P0/R/T*(a*2 + b*44 + c*16 + d*18 + e*40)/(a + b + c + d + e)*0.001
    # print(rohg,'rohg')

    return rohg

# 実際に解析を行うモデルの定義
@numba.jit("f8[:](f8,f8[:],f8[:],f8[:])",nopython=True)
def reaction(t,X,dX,params):
    # 定・変数定義
    T_in = params[5]
    Pa_in = params[0]*R*T_in
    Pb_in = params[1]*R*T_in
    Pc_in = params[2]*R*T_in
    Pd_in = params[3]*R*T_in
    Pe_in = params[4]*R*T_in
    T_jacket = params[6]
    u_in =params[7]
    void = params[8]
    dz = params[9]


    res = np.zeros(7*(NX))
    Ca = X[0:NX]
    Cb = X[NX:2*NX]
    Cc = X[2*NX:3*NX]
    Cd = X[3*NX:4*NX]
    Ce = X[4*NX:5*NX]
    T = X[5*NX:6*NX]
    u = X[6*NX:7*NX]
    P0 = Pa_in+Pb_in+Pc_in+Pd_in+Pe_in


    #初期条件
    res[0]     =  dX[0]
    res[NX]     =  dX[NX]
    res[2*NX]   =  dX[2*NX]
    res[3*NX]   =  dX[3*NX]
    res[4*NX]   =  dX[4*NX]
    res[5*NX]   = dX[5*NX]
    res[6*NX]   = u[0] - u_in

    for i in range(1,2):
        res[i]     = -void*dX[i]-(u[i]*Ca[i]-u[i-1]*Ca[i-1])/dz +void*Dz*(Ca[i+1]-Ca[i])/(dz**2)+ (1-void)*sc[0]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[NX+i]   = -void*dX[NX+i]-(u[i]*Cb[i]-u[i-1]*Cb[i-1])/dz +void*Dz*(Cb[i+1]-Cb[i])/(dz**2)+ (1-void)*sc[1]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[2*NX+i] = -void*dX[2*NX+i]-(u[i]*Cc[i]-u[i-1]*Cc[i-1])/dz +void*Dz*(Cc[i+1]-Cc[i])/(dz**2)+ (1-void)*sc[2]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[3*NX+i] = -void*dX[3*NX+i]-(u[i]*Cd[i]-u[i-1]*Cd[i-1])/dz +void*Dz*(Cd[i+1]-Cd[i])/(dz**2)+ (1-void)*sc[3]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[4*NX+i] = -void*dX[4*NX+i]-(u[i]*Ce[i]-u[i-1]*Ce[i-1])/dz +void*Dz*(Ce[i+1]-Ce[i])/(dz**2)+ (1-void)*sc[4]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[5*NX+i] = P0*void*(T[i]**(-2))*dX[5*NX+i]-u[i]*P0*(1/T[i]-1/T[i-1])/dz - P0/T[i]*(u[i]-u[i-1])/dz + void*Dz*P0*(1/T[i+1]-2/T[i]+1/T[i-1])/(dz**2) + (1-void)*R*(-2)*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[6*NX+i] = -(void*func_rohg(Ca[i],Cb[i],Cc[i],Cd[i],Ce[i],T[i],P0)*Cpg+(1-void)*rhos*Cps)*dX[5*NX+i]-func_rohg(Ca[i],Cb[i],Cc[i],Cd[i],Ce[i],T[i],P0)*Cpg*(T[i]*u[i]-T[i-1]*u[i-1])/dz + keff*(T[i+1]-2*T[i]+T[i-1])/(dz**2)+ (1-void)*(-Hr)*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18]) -2*U/dint*(T[i]-T_jacket)


    for i in range(2,NX-1):##上5式が成分収支式，次いで全物質収支式，エネルギー収支式
        res[i]     = -void*dX[i]-(u[i]*Ca[i]-u[i-1]*Ca[i-1])/dz +void*Dz*(Ca[i+1]-2*Ca[i]+Ca[i-1])/(dz**2)+ (1-void)*sc[0]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[NX+i]   = -void*dX[NX+i]-(u[i]*Cb[i]-u[i-1]*Cb[i-1])/dz +void*Dz*(Cb[i+1]-2*Cb[i]+Cb[i-1])/(dz**2)+ (1-void)*sc[1]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[2*NX+i] = -void*dX[2*NX+i]-(u[i]*Cc[i]-u[i-1]*Cc[i-1])/dz +void*Dz*(Cc[i+1]-2*Cc[i]+Cc[i-1])/(dz**2)+ (1-void)*sc[2]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[3*NX+i] = -void*dX[3*NX+i]-(u[i]*Cd[i]-u[i-1]*Cd[i-1])/dz +void*Dz*(Cd[i+1]-2*Cd[i]+Cd[i-1])/(dz**2)+ (1-void)*sc[3]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])
        res[4*NX+i] = -void*dX[4*NX+i]-(u[i]*Ce[i]-u[i-1]*Ce[i-1])/dz +void*Dz*(Ce[i+1]-2*Ce[i]+Ce[i-1])/(dz**2)+ (1-void)*sc[4]*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])


        #全物質収支式u固定T変動
        res[5*NX+i] = -u[i]*P0*(1/T[i]-1/T[i-1])/dz - P0/T[i]*(u[i]-u[i-1])/dz + void*Dz*P0*(1/T[i+1]-2/T[i]+1/T[i-1])/(dz**2) + (1-void)*R*(-2)*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18])

        #エネルギー収支式T固定u変動
        res[6*NX+i] = -0.1*(void*func_rohg(Ca[i],Cb[i],Cc[i],Cd[i],Ce[i],T[i],P0)*Cpg+(1-void)*rhos*Cps)*dX[5*NX+i]-func_rohg(Ca[i],Cb[i],Cc[i],Cd[i],Ce[i],T[i],P0)*Cpg*(T[i]*u[i]-T[i-1]*u[i-1])/dz + keff*(T[i+1]-2*T[i]+T[i-1])/(dz**2)+ (1-void)*(-Hr)*func_rCH4(T[i],Ca[i],Cb[i],Cc[i],Cd[i],params[10:18]) -2*U/dint*(T[i]-T_jacket)


    #境界条件
    for i in range(NX-1,NX):
        res[i] = Ca[i] - Ca[i-1]
        res[2*i+1] = Cb[i] - Cb[i-1]
        res[3*i+2] = Cc[i] - Cc[i-1]
        res[4*i+3] = Cd[i] - Cd[i-1]
        res[5*i+4] = Ce[i] - Ce[i-1]
        res[6*i+5] = u[i] - u[i-1]
        res[7*i+6] = T[i] - T[i-1]

    return res

errorbox = [['params','i']]

# パラメータを引数にして，モデルを解き，出口での変数を返すコード
def my_model(params):
    pr = params

    # 変数を格納するための空のリスト
    Xa = []
    Xb = []
    Xc = []
    Xd = []
    Xe = []
    Fa = []
    Fb = []
    Fc = []
    Fd = []
    Fe = []
    t0 = 0

    # 実験条件数分だけ繰り返す
    for i in range(0,n_data):
        yd0 = np.zeros(7*NX)
        t0 = 0
        p0 = (Ca_in[i],Cb_in[i],Cc_in[i],Cd_in[i],Ce_in[i],T_in[i],T_jacket[i],u_in[i],void[i],reactorlength[i]/(NX-1),pr[0],pr[1],pr[2],pr[3],pr[4],pr[5],pr[6],pr[7])
        P_total = (p0[0]+p0[1]+p0[2]+p0[3]+p0[4])*R*p0[5] # 非ゲージ圧力,圧力一定なので初期圧力まんまで良いS

        model = Implicit_Problem(reaction, guess[i], yd0, t0,p0=p0) #Create an Assimulo problem
        # model.name = 'tubular' #Specifies the name of problem (optional)
        imp_sim = IDA(model)

        #Sets the paramters
        # 正直勉強不足でよくわかっていない部分，usesensとかは使ってしまうと勝手にパラメータの調整が入るらしく良くないっぽい
        imp_sim.algvar = li
        # imp_sim.atol = atol
        imp_sim.usesens = False
        imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test
        # imp_sim.report_continuously = False #Store data continuous during the simulationこ
        # imp_sim.pbar = p0
        # imp_sim.suppress_sens = False          #Dont suppress the sensitivity variables in the error test.
        # imp_sim.display_progress = False
        # imp_sim.backward = False
        # imp_sim.store_event_points = False
        imp_sim.verbosity = 50 # 結果を非表示
        # imp_sim.verbosity = 30 # 結果を表示

        #Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
        imp_sim.re_init(t0,guess[i],yd0)
        imp_sim.make_consistent('IDA_YA_YDP_INIT')

        tfinal = 75  #Specify the final time
        ncp = 10    #Number of communication points (number of return points)


        # Assimuloは定期的にエラーを吐いてくるので，try-exceptでPOWERして，
        # エラー処理を行う（エラーを吐いたら返り値として-10000とかいう頭の悪い誤差を与えて，尤度を低くしてもらうことでその粒子は採択されなくなる）
        try:
            # ここでシミュレーションの実行，返り値としてt,y,ydが得られる．それぞれ時間，変数，変数の時間微分
            t, y, yd = imp_sim.simulate(tfinal, ncp)


            # ここからは得られた結果について，出口部分かつ最終的な値を抽出してリストに格納していく

            # 標準状態での流量に換算する
            Fa.append(y[-1][-6*NX-1]*S*y[-1][-1]*60*R*y[-1][-NX-1]/(P_total)*10**6*(P_total)/P_stp*298/y[-1][-NX-1])
            Fb.append(y[-1][-5*NX-1]*S*y[-1][-1]*60*R*y[-1][-NX-1]/(P_total)*10**6*(P_total)/P_stp*298/y[-1][-NX-1])
            Fc.append(y[-1][-4*NX-1]*S*y[-1][-1]*60*R*y[-1][-NX-1]/(P_total)*10**6*(P_total)/P_stp*298/y[-1][-NX-1])
            Fd.append(y[-1][-3*NX-1]*S*y[-1][-1]*60*R*y[-1][-NX-1]/(P_total)*10**6*(P_total)/P_stp*298/y[-1][-NX-1])
            Fe.append(y[-1][-2*NX-1]*S*y[-1][-1]*60*R*y[-1][-NX-1]/(P_total)*10**6*(P_total)/P_stp*298/y[-1][-NX-1])

            # そのまんま濃度を受け取る
            # Ca.append(y[-1][-6*NX-1])
            # Cb.append(y[-1][-5*NX-1])
            # Cc.append(y[-1][-4*NX-1])
            # Cd.append(y[-1][-3*NX-1])
            # Ce.append(y[-1][-2*NX-1])
            # T.append(y[-1][-n-1])
            # u.append(y[-1][-1])

            # モル分率にする
            molfraction_a = y[-1][-6*NX-1]/(y[-1][-6*NX-1]+y[-1][-5*NX-1]+y[-1][-4*NX-1]+y[-1][-3*NX-1]+y[-1][-2*NX-1])
            molfraction_b = y[-1][-5*NX-1]/(y[-1][-6*NX-1]+y[-1][-5*NX-1]+y[-1][-4*NX-1]+y[-1][-3*NX-1]+y[-1][-2*NX-1])
            molfraction_c = y[-1][-4*NX-1]/(y[-1][-6*NX-1]+y[-1][-5*NX-1]+y[-1][-4*NX-1]+y[-1][-3*NX-1]+y[-1][-2*NX-1])
            molfraction_d = y[-1][-3*NX-1]/(y[-1][-6*NX-1]+y[-1][-5*NX-1]+y[-1][-4*NX-1]+y[-1][-3*NX-1]+y[-1][-2*NX-1])
            molfraction_e = y[-1][-2*NX-1]/(y[-1][-6*NX-1]+y[-1][-5*NX-1]+y[-1][-4*NX-1]+y[-1][-3*NX-1]+y[-1][-2*NX-1])
            Xa.append(molfraction_a)
            Xb.append(molfraction_b)
            Xc.append(molfraction_c)
            Xd.append(molfraction_d)
            Xe.append(molfraction_e)

            # 温度と速度は尤度計算に使わないのでコメントアウト
            # T.append(y[-1][-NX-1])
            # u.append(y[-1][-1])
        except:
            # # Assimuloがエラーを吐いた場合の処理
            # print('Assimuloでエラーが発生しました。。。')

            # エラーを吐いたパラメータとその実験条件をerroboxに格納する
            error_ = [params,i]
            errorbox.append(error_)

            # エラーを吐いたら返り値として-10000とかいう頭の悪い誤差を与えて，尤度を低くしてもらうことでその粒子は採択されなくなる.
            # モル分率としては0を与える
            y = np.ones([5,7*NX])*-10000
            Fa.append(y[-1][-6*NX-1])
            Fb.append(y[-1][-5*NX-1])
            Fc.append(y[-1][-4*NX-1])
            Fd.append(y[-1][-3*NX-1])
            Fe.append(y[-1][-2*NX-1])
            Xa.append(0)
            Xb.append(0)
            Xc.append(0)
            Xd.append(0)
            Xe.append(0)
        
        # 一応モデルを消す操作（たぶん意味無い）
        del imp_sim
        del model

    # モル分率
    molfraction = np.ones([5,n_data])
    molfraction[0,:] = Xa
    molfraction[1,:] = Xb
    molfraction[2,:] = Xc
    molfraction[3,:] = Xd
    molfraction[4,:] = Xe

    # 流量
    Flow = np.ones([5,n_data])
    Flow[0,:] = Fa
    Flow[1,:] = Fb
    Flow[2,:] = Fc
    Flow[3,:] = Fd
    Flow[4,:] = Fe

    # 返り値として，流量，モル分率，errorbox
    return Flow,molfraction

# 対数尤度の計算
def my_loglike(y, data, sigma, n_data, scale=1.0):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """

    # 対数尤度を格納する空のリスト
    log = []

    # 各成分について尤度を計算
    for i in range(0,5):
        log_l =  (y[i,:]-data[i,:])*(y[i,:]-data[i,:])
        log_l = np.sum(log_l.T, axis=0)
        log_l = -(0.5 /sigma**2) * log_l - n_data * np.log(sigma)

        # 各成分の対数尤度をリストに格納
        log.append(log_l)

    # 尤度をすべて足し合わせる
    output = np.sum(log)

    return output

# 並列計算を行う際に直接呼び出す部分
@ray.remote
def cal_parallel_new(params):
    # パラメータを反応速度式のパラメータとσに分ける
    params0 = params[:-1]

    # σを推定しない場合，sigmaはsigma_trueに固定される
    if est_siguma:
        sigma = params[-1]
    else:
        sigma = sigma_true

    # パラメータを与え，それによって返ってくる各実験条件での返り値を得る
    ycal1,molfraction = my_model( params0)
    # スケーリング,とくに必要ない
    y_scale = 1
    ycal = ycal1 * y_scale

    # 対数尤度を計算する
    lk = my_loglike(ycal1, data, sigma, n_data, scale=1.0)

    # 返り値に尤度，モル分率，エラー
    return lk,molfraction

# 事前分布の計算？
def cal_prior(theta):
    # σの分を抜く    
    # for i in range(n_state-1):
    # print(theta)
    if normal_pred:
        # 全体のリスト
        all_values = list(range(8))
        # 残りの値
        remaining_values = [x for x in all_values if x not in prior_list_uni]

        if taylor:
            params_p = np.zeros([num_est_params,n_particle])
            for i in remaining_values:
                p = scipy.stats.norm.pdf(theta[:,i],trueparams[i],np.abs(trueparams[i])*coefficent[i])
                # print(p)
                params_p[i,:] = p
                # print(params_p)
            for i in prior_list_uni:
                dl = high_limit[i] - low_limit[i]
                p = scipy.stats.uniform.pdf(theta[:,i], low_limit[i], dl)
                # print(p)
                params_p[i,:] = p
            # print(params_p)
            result = np.prod(params_p.T, axis=1)
            

        else:
            params_p = np.zeros([num_est_params-1,n_particle])
            for i in range(num_est_params-1):
                p = scipy.stats.norm.pdf(theta[:,i],trueparams[i],np.abs(trueparams[i])*coefficent[i])
                # print(p)
                params_p[i,:] = p
                # print(params_p)
            result = np.prod(params_p, axis=0)
    else:
        dl = high_limit_array - low_limit_array
        p = scipy.stats.uniform.pdf(theta, [low_limit[i] for i in est_position], dl)
        result = np.prod(p.T, axis=0)
    # print(result)
    return result

# 粒子数の表示（推定開始時に変な条件で処理していないか確認するため）
print(f"粒子数{n_particle},実験条件数{n_data}")

# errror格納用リスト
errorboxlist = []


# 初期値の設定
guess = np.zeros([n_data,7*NX])
for i in range(0,n_data):
    guess0 = np.ones(7*NX)
    guess0[0:NX] = Ca_in[i]
    guess0[NX:2*NX] = Cb_in[i]
    guess0[2*NX:3*NX] = Cc_in[i]
    guess0[3*NX:4*NX] = Cd_in[i]
    guess0[4*NX:5*NX] = Ce_in[i]
    guess0[5*NX:6*NX] = T_in[i]
    guess0[5*NX+1:6*NX] = 400
    guess0[6*NX:7*NX] = u_in[i]
    guess[i,:] = guess0

"""
Real experimental data
"""
# print('realdata')
# y_scale = 1.0
# data = np.ones([5,datapoint])
# data[0,:] = Fa_out
# data[1,:] = Fb_out
# data[2,:] = Fc_out
# data[3,:] = Fd_out
# data[4,:] = Fe_out
# n_data = np.size(data[0,:])
# print(n_data)

# # Stores mole fraction of real data in data_mol
# data_mol = np.ones([5,datapoint])
# data_mol[0,:] = Xa_out
# data_mol[1,:] = Xb_out
# data_mol[2,:] = Xc_out
# data_mol[3,:] = Xd_out
# data_mol[4,:] = Xe_out



"""
Artificially generated data
"""
print('artificial data')

# Flow is flow rate, molfraction is mole fraction
Flow,molfraction = my_model(trueparams)
data_mol = molfraction
data = Flow

# Add noise to the data
for i in range(0,5):
    data[i,:] = 1.0 * sigma_true * np.random.standard_normal(n_data) + data[i,:]
for i in range(n_data):
    data_mol[0,i] = data_mol[0,i]/np.sum([data_mol[0,i],data_mol[1,i],data_mol[2,i],data_mol[3,i],data_mol[4,i]])
    data_mol[1,i] = data_mol[1,i]/np.sum([data_mol[0,i],data_mol[1,i],data_mol[2,i],data_mol[3,i],data_mol[4,i]])
    data_mol[2,i] = data_mol[2,i]/np.sum([data_mol[0,i],data_mol[1,i],data_mol[2,i],data_mol[3,i],data_mol[4,i]])
    data_mol[3,i] = data_mol[3,i]/np.sum([data_mol[0,i],data_mol[1,i],data_mol[2,i],data_mol[3,i],data_mol[4,i]])
    data_mol[4,i] = data_mol[4,i]/np.sum([data_mol[0,i],data_mol[1,i],data_mol[2,i],data_mol[3,i],data_mol[4,i]])

# Save the generated data as CSV files
df_data = pd.DataFrame(data)
df_data_mol = pd.DataFrame(data_mol)
df_data.to_csv("data.csv", index=False, header=False)
df_data_mol.to_csv("data_mol.csv", index=False, header=False)

# 粒子毎の並列処理
# @profile()
def sim_particle(particle, guess):
    print('sim_particle')
    # メモリー使用量を参照して，並列化を再initするか決定する
    mem = psutil.virtual_memory() 
    if mem.percent > 80:
        print(mem)
        ray.shutdown()
        time.sleep(10)
        ray.init(num_cpus=n_cores,ignore_reinit_error=True)

    p_pred_trues[:, est_position] = particle
    

    # 並列化計算を実行する
    results = [cal_parallel_new.remote(p_pred_trues[i,:]) for i in range(n_particle)]

    # 並列計算の結果を取得する
    llk_Cl = ray.get(results)

    # 取得した結果を使える形に成型する
    llk, C_l_ = zip(*llk_Cl)

    return llk,C_l_


# parity plotを描画，保存する
def ChromatogramDrawerWhileSMC(dirname01_,dirname02,C, name_):
    tag = ["Xa","Xb","Xc","Xd","Xe","T","u"]
    for i in range(0,5):

        # 箱ひげ図にするためのデータ成型
        Clist = []
        ylist = []
        for k in range(n_data):
            CL = [arr[i][k] for arr in C]
            ylist.append(F[i,k])
            Clist.append(CL)

        # 箱ひげ図のparity plot
        plt.figure(figsize=(7, 7))
        plt.ylabel(f"simulation{tag[i]}[-]")
        plt.xlabel(f"data{tag[i]}[-]")
        x_values = [0, 1]  # x軸の範囲
        y_values = [0, 1]  # y軸の範囲
        # 対角線をプロットします
        plt.plot(x_values, y_values, 'r--')
        # vert = Trueで縦向き
        plt.boxplot(Clist, positions=ylist, vert = True,showfliers=False, widths = 0.01)
        plt.xlim([-0.05,1])
        plt.ylim([-0.05,1])
        plt.yticks([0,0.2,0.4,0.6,0.8,1])
        plt.xticks([])
        # plt.yticks([0,0.2,0.4,0.6,0.8,1])
        plt.savefig(f"{dirname01_}Overlayed_Simulation_while_SMC_{name_}_N_{i}.png", bbox_inches="tight", dpi=300)
        plt.close()

        # 平均値とデータのparity plot
        plt.figure(figsize=(7, 7))
        plt.ylabel(f"simulation{tag[i]}[-]")
        plt.xlabel(f"data{tag[i]}[-]")
        x_values = [0, 1]  # x軸の範囲
        y_values = [0, 1]  # y軸の範囲
        # 対角線をプロットします
        plt.plot(x_values, y_values, 'r--')
        plt.boxplot(Clist, positions=ylist, vert = True, widths = 0.01, showmeans=True,showbox=False,showcaps=False,showfliers=False,meanprops=dict(marker='o'),whis = [25,75],sym='')
        plt.xlim([-0.05,1])
        plt.ylim([-0.05,1])
        plt.xticks([0,0.2,0.4,0.6,0.8,1])
        plt.yticks([])
        plt.savefig(f"{dirname02}Overlayed_Simulation_while_SMC_{name_}_N_{i}.png", bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == '__main__':
    print('start')

    # 初期分布を入れる空の配列
    p_pred = np.zeros((n_particle, num_est_params))

    # フィルター後の分布を入れるための空の配列
    p_filt = np.zeros((n_particle, num_est_params))

    # 最初はどの粒子も同じ重みを持つので、それぞれに1/n_particleの重みを設定（重みの総和は1）
    p_weight = np.ones(n_particle) / n_particle

    # Resamplingで用いる
    p_is = np.zeros(n_particle, dtype=int)

    # 粒子数×尤度計算の際に使うデータの数（例えば各実験条件に対して、各成分の出口濃度を計算に使うなら、実験条件数がn_dataに充たる）
    y_cal = np.zeros((n_particle, n_data))

    # 尤度関係の何かを格納する（粒子数分だけ要素がある）
    d_lk = np.zeros(n_particle)
    # 尤度関係の何かを格納する（粒子数分だけ要素がある）
    lk1 = np.zeros(n_particle)

    # 真値だけ入った配列
    p_pred_trues = np.tile(np.append(trueparams, sigma_true), (n_particle, 1))
    # print(p_pred_trues)


    start_time = time.time()

    # normal_pred= Trueのときは事前分布が正規分布になる
    # 事前分布を一様分布として設定
    counter = 0
    i = 0
    # print(num_est_params)
    while counter < num_est_params:
        if est_params_list[i] == 1:
            p_pred[:,counter] = np.random.uniform(low_limit[i], high_limit[i], n_particle)
            counter += 1
        i += 1
        # print(i,counter)

    est_position = [i for i, x in enumerate(est_params_list) if x == 1]
    # print(est_position)
    est_position_set = set(est_position)
    uni_list_set = set(uni_list)
    def_set = uni_list_set - est_position_set

    if normal_pred:
        # 事前分布として正規分布を与える
        print('normal_distribution')

        # trueparamsを中心として，Coefficentで定義された係数倍を分散として持つ正規分布を与える
        counter = 0
        while counter== num_est_params:
            if num_est_params[i] == 1:
                p_pred[:,i] = np.random.normal(trueparams[i], np.abs(trueparams[i])*coefficent[i], n_particle)
            counter += 1

        if taylor:
            # 新規分布を事前分布として与え，かついくつかの分布を一様分布にするときに用いる
            # いじる部分の数と推定パラメータ数の大きさを比べたり、いじる部分の位置とuni_listの数が等しくなれば、、、みたいな処理
            print('taylor')
            if len(def_set):
                sys.exit()
                print('推定するパラメータ数に対してtaylorしている値が多いです。パラメータを確認してください。')
            else: 
                sys.exit()
                counter_taylor = 0
                j = 0
                while counter_taylor == num_est_params:
                    if est_position[j] == uni_list[counter_taylor]:
                        p_pred[:,counter] = np.random.uniform(low_limit[counter_taylor], high_limit[counter_taylor], n_particle)
                        j += 1
                    counter_taylor += 1
    high_limit_array = np.array([high_limit[i] for i in est_position])
    low_limit_array = np.array([low_limit[i] for i in est_position])


    firstpred = f'{dirnamepred}first_p_pred.csv'
    np.savetxt(firstpred, p_pred, delimiter=',')

    # 事前分布をグラフに
    # このヒストグラムだけ関数にしていないので，そうしても良いかも
    fig = plt.figure(figsize=(10, 14))
    columns_origin=["Af","Eaf","Ar","Ear","BCO2","dHCO2","BH2O","dHH2O","sigma"]

    # print(est_position_set)

    sub = 0
    for index in est_position:
        ax1 = plt.subplot(num_est_params, 1, sub+1)
        ax1.hist(p_pred[:, sub], 40, range=(low_limit[index], high_limit[index]), density=True)
        # ax1.axvline(p_filt[:, index].mean(),color = 'red', linestyle = 'dashed', linewidth =1)
        # 平均値をグラフのどこかに書き込みたかったが失敗，Excelで簡単に出力できるのでよしとする
        # ax1.text(0,0,f'Mean:{p_filt[:, index].mean()}',va = 'center', ha ='center', transform=ax1.transAxes)　
        ax1.set_ylabel(columns_origin[index])
        ax1.grid('on')
        sub += 1
    plt.savefig(f"{dirnameProgress}00_PriorDistribution.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 最初のinit（並列展開）
    ray.init(num_cpus=n_cores,ignore_reinit_error=True)
    RAY_DEDUP_LOGS=0

    try:

        # 初期分布の状態で尤度推定＆parity plot用のデータ取得
        lk,C_l_= sim_particle(p_pred,guess)
        print('first_step')

        # # errorの格納
        # errorboxlist.append(errors)

        time1 = time.time()

        # parity plotを描く
        ChromatogramDrawerWhileSMC(dirnametubular,dirnametubular_mean, C_l_, f"00_PriorDistribution")

        gamma_old = 0.0
        gamma_new = 1.0

        # 繰り返し計算（上限あり）
        for step in range(1, itr_max):
            # 新しいγを定義
            gamma_new = gamma_old + d_gamma_max
            if gamma_new > 1.0:
                gamma_new = 1.0

            # 最大の尤度を選択
            max_lk = np.max(lk)
            # 各種尤度-最大尤度の配列を作る
            d_lk = lk - max_lk
            # gm_reduction_itrの回数分（途中で終わるかも）繰り返し(reduction means 削減)
            for i in range(gm_reduction_itr):
                gm = gamma_new - gamma_old

                # 尤度差分とγをかけ合わせた値を指数として、ネイピア数を乗算して重み配列として配列化（尤度自体が対数尤度なので、元に戻している？）
                p_weight = np.exp(d_lk * gm)

                # 得られた重みを足し合わせ
                sum_weight = np.sum(p_weight)

                # 合計重みで割ることで規格化する
                p_weight = p_weight / sum_weight

                # 有効粒子採択数を計算
                ess = np.sum(p_weight ** 2)
                ess = 1.0 / ess / n_particle

                if ess > ess_limit:
                    print(f'ess>ess_limit:{ess}')
                    break

                # 新たなγを生成（今ステップでのγ増加量×減少率＋旧γ）
                gamma_new = (gamma_new - gamma_old) * gm_reduction_rate + gamma_old

            if ess < ess_limit:
                print("ess reduction warning: ess = ", ess)

            # p_is に重みと粒子をかけ合わせ、少数部分を切り捨てたものを格納
            p_is = np.trunc(p_weight * n_particle).astype(int)

            # p_weightにp_weight-p_is*inv_Np(1/n_particle)をかけ合わせたものを格納
            p_weight = p_weight - p_is * inv_Np

            # n_tmpに粒子数-p_isの合計を格納
            n_tmp = n_particle - np.sum(p_is)

            #wrandに0以上1未満の乱数×(1/n_particle)を格納
            wrand = np.random.rand() * inv_Np

            print('n_tmp:',n_tmp)
            sum = 0.0
            n = 0

            #p_pred_copyにp_predを明示的にコピー
            p_pred_copy = p_pred.copy()

            for j in range(n_particle):
                # sumに現在対象となっている粒子の重みを足し合わせている
                sum += p_weight[j]
                if (sum >= wrand):
                    # 現在sumがwrand以上に
                    # なっているのなら、p_isの現在対象となっている粒子の部分について+1
                    p_is[j] += 1

                    # wrandに1/n_particleを足す
                    wrand += inv_Np

                    # n_tmpから1を引く
                    n_tmp -= 1
                for k in range(p_is[j]):
                    # p_predのn番目に現在対象となっている粒子のp_predをコピーする
                    p_filt[n, :] = p_pred_copy[j, :]

                    # lk1のn番目は現在の粒子の尤度を入れる、その後nを1増やす
                    lk1[n] = lk[j]
                    n += 1

            # r_acの定義（要素数はn_particleのゼロ配列）
            r_ac = np.zeros(n_particle)
            
            # mhstep比を1にする
            mhstep_ratio = 1.0
            # p_pred_storage = p_filt.copy()

            if gamma_new >= 1.0:
                print(f'gamma_new>=1.0:{gamma_new}')
                # γが1以上ならnMHはad_mhstep_num
                nMH = ad_mhstep_num
                # rateはmhstep_factor
                rate = mhstep_factor
                # r_thはr_threshold_f
                r_th = r_threshold_f
            else:
                print(f'gamma_new<1.0:{gamma_new}')
                # それ以外ならnMHはmhstep_num
                nMH = mhstep_num
                # rateは同じ
                rate = mhstep_factor
                # r_thにはr_threshold
                r_th = r_threshold
            for j in range(nMH):

                # p_filtの転置行列について共分散行列を作成
                cov_m = np.cov(p_filt.T, bias=True)

                # csv_mに共分散行列とw_covの積を入れる
                cov_m = cov_m * w_cov

                # np.random.multivariate_normal()は多変量正規分布に従う乱数を生成
                #np.zeros(n_state)は平均ベクトルとして要素が全てゼロの長さn_stateの配列を作成します。これにcov_mを共分散行列として与え、n_particle個の乱数ベクトルを生成します。
                #そして、これらの乱数ベクトルをmhstep_ratio倍してp_filtに足してp_predを計算します。結果のp_predは、与えられたp_filtに多変量正規分布から生成されたノイズを加えた予測値の行列になります。
                p_pred = p_filt + np.random.multivariate_normal(np.zeros(num_est_params), cov_m, n_particle) * mhstep_ratio


                if normal_pred:
                    if taylor:
                        # 現在の分布について、設定した分布の中心(trueprams)を平均値とした正規分布においての確率を計算(正規分布部分については)。
                        # 一様分布部分については、範囲内に入っているのか確認。
                        p0_1 = cal_prior(p_filt)
                        p0_2 = cal_prior(p_pred)
                        print('p0_2',p0_2)

                        # 正規分布については範囲外のやつは過去の値に入れなくてはいけないのだ
                        p0 = np.int32(p0_2 > 0)
                        # print(p0)
                        p_pred = p_pred * p0[:, None] + p_filt * (1.0 - p0[:, None])

                        # 提案粒子群について尤度計算
                        lk2,C_l_ = sim_particle(p_pred, guess)
                        # errorboxlist.append(errors)
                        px = lk2 - lk1

                        # # 正規、一様調分布を処理するためのリスト制作
                        # all_values = list(range(9))
                        # # 残りの値
                        # remaining_values = [x for x in all_values if x not in uni_list]

                        pp = np.zeros([num_est_params,n_particle])

                        pp = np.exp(px * gamma_new) * (p0_2/p0_1)* p0

                        # for i in uni_list:
                        #     # p0を掛けることで、範囲外に出て前の値を複製したやつは採択されないようにする
                        #     pp[i,:] = np.exp(px[i,:] * gamma_new) * p0[i,:]

                        # for i in remaining_values:
                        #     pp[i,:] = np.exp(px[i,:] * gamma_new) * (p0_2[i,:]/p0_1[i,:])

                        rr = np.random.uniform(0, 1, n_particle)
                        r = np.int32(pp >= rr)
                        # 採択された奴はp_pred、採択されなかった奴はp_filtを「引き継ぐ
                        p_filt = p_pred * r[:, None] + p_filt * (1.0 - r[:, None])
                        lk1 = lk2 * r + lk1 * (1.0 - r)
                        r_ac = np.maximum(r_ac, r)
                    else:
                        p0_1 = cal_prior(p_filt)
                        p0_2 = cal_prior(p_pred)
                        # p0 = np.int32(p0_2 > 0)
                        # p_pred = p_pred * p0[:, None] + p_filt * (1.0 - p0[:, None])

                        lk2,C_l_ = sim_particle(p_pred, guess)
                        # errorboxlist.append(errors)
                        px = lk2 - lk1

                        pp = np.exp(px * gamma_new) * (p0_2/p0_1)

                        rr = np.random.uniform(0, 1, n_particle)
                        r = np.int32(pp >= rr)
                        p_filt = p_pred * r[:, None] + p_filt * (1.0 - r[:, None])
                        lk1 = lk2 * r + lk1 * (1.0 - r)
                        r_ac = np.maximum(r_ac, r)
                else:
                    p0_1 = cal_prior(p_filt)
                    p0_2 = cal_prior(p_pred)
                    # print('p0_2',p0_2)

                    p0 = np.int32(p0_2 > 0)
                    # print(p0)

                    p_pred = p_pred * p0[:, None] + p_filt * (1.0 - p0[:, None])
                    lk2,C_l_ = sim_particle(p_pred, guess)
                    # errorboxlist.append(errors)
                    px = lk2 - lk1

                    pp = np.exp(px * gamma_new) * p0

                    rr = np.random.uniform(0, 1, n_particle)
                    r = np.int32(pp >= rr)

                    p_filt = p_pred * r[:, None] + p_filt * (1.0 - r[:, None])

                    lk1 = lk2 * r + lk1 * (1.0 - r)
                    r_ac = np.maximum(r_ac, r)

                if r_ac.sum() > r_th * n_particle:
                    print(f'r_ac.sum() > r_th * n_particle:{r_ac.sum()}')
                    break

                if r_ac.sum() < r_threshold_min * n_particle:
                    mhstep_ratio = mhstep_ratio * 0.5
                    print(mhstep_ratio)

            p_pred = p_filt.copy()
            lk = lk1.copy()

            print(f"iteration:{step}, nMH:{j}, Calculation time:{time.time() - start_time}, ESS:{ess}, Max Likelihood:{max_lk}, New Gamma:{gamma_new}, Number of Adoption:{r_ac.sum()}")
            # print(C_l_)
            # np.savetxt(f'{dirnamevaliables}/valiables{step}.csv',C_l_,delimiter=',')

            # ray.shutdown()
            if gamma_new == 1.0: 
                print(f'gamma_new:{gamma_new}')
                break
            gamma_old = gamma_new
            # ここまでが1ループ、あとは描画

            ChromatogramDrawerWhileSMC(dirnametubular,dirnametubular_mean, C_l_,f"step=0{step}_nMH=0{j}")
            np.savetxt(f'{dirnamepred}{step}_p_pred.csv', p_pred, delimiter=',')
            # print(C_l_)
            # np.savetxt(f'{dirnameC_l_}{step}_C_l_.csv', C_l_, delimiter=',')

            fig = plt.figure(figsize=(10, 14))
            columns_origin=["Af","Eaf","Ar","Ear","BCO2","dHCO2","BH2O","dHH2O","sigma"]
            sub = 0
            for index in est_position:
                ax1 = plt.subplot(num_est_params, 1, sub+1)
                ax1.hist(p_filt[:, sub], n_hist, range=(low_limit[index], high_limit[index]), density=True)
                ax1.axvline(p_filt[:, sub].mean(),color = 'red', linestyle = 'dashed', linewidth =1)
                # 平均値をグラフのどこかに書き込みたかったが失敗，Excelで簡単に出力できるのでよしとする
                ax1.axvline(trueparams_withsigma[index], color = 'black', linewidth =2)
                # ax1.text(p_filt[:, index].mean()*1.1,0.5,'Mean:{:,2f}'.format(p_filt[:, index].mean()))
                ax1.grid('on')
                ax1.set_ylabel(columns_origin[index])
                sub += 1
            plt.savefig(f"{dirnameProgress}SMC_Histgram_Progress_step=0{step}_nMH=0{j}.png", bbox_inches="tight", dpi=300)
            plt.close()


        if gamma_new < 1.0:
            print("tempering does't complete: last gamma =", gamma_new)



        end_time = time.time() - start_time
        print(end_time)
        columns_origin=["Af","Eaf","Ar","Ear","BCO2","dHCO2","BH2O","dHH2O","sigma"]
        fig = plt.figure(figsize=(10, 14))
        sub = 0
        #    t=np.arange(0,T+1)
        for index in est_position:
            ax1 = plt.subplot(num_est_params, 1, sub+1)
            ax1.hist(p_filt[:, sub], n_hist, density=True)
            ax1.axvline(p_filt[:, sub].mean(),color = 'red', linestyle = 'dashed', linewidth =1)
            # 平均値をグラフのどこかに書き込みたかったが失敗，Excelで簡単に出力できるのでよしとする
            # ax1.text(p_filt[:, index].mean()*1.1,0.5,'Mean:{:,2f}'.format(p_filt[:, index].mean()))
            ax1.axvline(trueparams_withsigma[index], color = 'black', linewidth =2)
            ax1.grid('on')
            ax1.set_ylabel(columns_origin[index])
            sub += 1
        plt.savefig(f"{dirname}Posterior_Distribution.png", bbox_inches="tight", dpi=300)
        plt.close()

        lim = 1
        fig = plt.figure(figsize=(10, 14))
        sub = 0
        for index in est_position:
            ax1 = plt.subplot(num_est_params, 1, sub+1)
            ax1.xlim = (-lim, lim)
            ax1.hist(p_filt[:, sub], 40, density=True)
            ax1.grid('on')
            sub += 1
        #    t=np.arange(0,T+1)
        df = pd.DataFrame(p_filt)
        labels = [columns_origin[i] for i in est_position]
        pairplot = sns.pairplot(df, corner=True)
        pairplot.x_vars = labels
        pairplot.y_vars = labels
        # pairplot._add_axis_labels()
        plt.savefig(f"{dirname}Pairplot.png", bbox_inches="tight", dpi=300)
        plt.close()
        columns_origin=["Af","Eaf","Ar","Ear","BCO2","dHCO2","BH2O","dHH2O","sigma"]
        columns = []
        for i in range(num_est_params-1):
            columns.append(columns_origin[i])
        columns.append('sigma')
        # df_Posterior = pd.DataFrame(p_filt, columns=["Ha [-]", "Hb [-]", "ka [1/s]", "kb [1/s]", "ba [vol%^-1]", "bb [vol%^-1]"])
        # print(p_filt,columns,p_pred)
        df_Posterior = pd.DataFrame(p_filt, columns = columns)#,"Ar","Ear","BCO2","dHCO2","BH2O","dHH2O"])
        df_Posterior.to_csv(f"{dirname}Posterior_Distribution.csv", index=False)

        last_pred = f'{dirnamepred}last_p_pred.csv'
        np.savetxt(last_pred, p_filt, delimiter=',')




        # 事前と事後を重ねて描く

        # 参照するパラメータを読み込む
        p_pred1 = pd.read_csv(firstpred, header=None).values
        p_pred2 = pd.read_csv(last_pred, header=None).values

        min_list = []
        max_list = []
        for i in range(num_est_params):
            min1 = min(p_pred1[:,i])
            min2 = min(p_pred2[:,i])
            max1 = max(p_pred1[:,i])
            max2 = max(p_pred2[:,i])
            if min1<min2:
                min_list.append(min1)
            else:
                min_list.append(min2)
            if max1>max2:
                max_list.append(max1)
            else:
                max_list.append(max2)

        fig = plt.figure(figsize=(10, 14))
        print('compare_hist')
        columns_origin=["Af","Eaf","Ar","Ear","BCO2","dHCO2","BH2O","dHH2O","sigma"]
        sub = 0
        for index in est_position:
            ax1 = plt.subplot(num_est_params, 1, sub+1)
            ax1.hist(p_pred1[:, sub], n_hist, range=(min_list[sub], max_list[sub]), density=True, color = (0,0,1,0.3))
            ax1.axvline(p_pred1[:, sub].mean(),color = 'blue', linestyle = 'dashed', linewidth =1)
            ax1.hist(p_pred2[:, sub], n_hist, range=(min_list[sub], max_list[sub]), density=True, color = (1,0,0,0.7))
            ax1.axvline(p_pred2[:, sub].mean(),color = 'purple', linestyle = 'dashed', linewidth =1)
            ax1.axvline(trueparams_withsigma[index], color = 'black', linewidth =2)
            ax1.grid('on')
            ax1.set_ylabel(columns_origin[index])
            sub += 1
        plt.savefig(f"{dirname}Histgram_compare.png", bbox_inches="tight", dpi=300)
        plt.close()
    except Exception as e:
        # 発生中の例外に関する情報を取得する
        etype, value, tb = sys.exc_info()
        
        traceback.format_exception(etype, value, tb) # list[str] を返す
        traceback.format_tb(tb) # list[str] を返す
        traceback.format_exception_only(etype, value) # list[str] を返す
        traceback.format_exc() # tuple[str] を返す
        print(traceback.format_exception(etype, value, tb))
        print(f"{e.__class__.__name__}: {e}")
