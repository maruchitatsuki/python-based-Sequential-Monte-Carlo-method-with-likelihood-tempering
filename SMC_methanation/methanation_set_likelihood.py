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
from methanation_set_conditon import *

# === Profiling ===
from memory_profiler import profile


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
def my_model(params, initial_guess):
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

        model = Implicit_Problem(reaction, initial_guess[i], yd0, t0,p0=p0) #Create an Assimulo problem
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
        imp_sim.re_init(t0,initial_guess[i],yd0)
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