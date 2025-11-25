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
from methanation_set_likelihood import *

# === Profiling ===
from memory_profiler import profile

# 並列計算を行う際に直接呼び出す部分
@ray.remote
def cal_parallel_new(params, obs_data, initial_guess):
    # パラメータを反応速度式のパラメータとσに分ける
    params0 = params[:-1]

    # σを推定しない場合，sigmaはsigma_trueに固定される
    if est_sigma:
        sigma = params[-1]
    else:
        sigma = sigma_true

    # パラメータを与え，それによって返ってくる各実験条件での返り値を得る
    ycal1,molfraction = my_model(params, initial_guess)
    # スケーリング,とくに必要ない
    y_scale = 1
    ycal = ycal1 * y_scale

    # 対数尤度を計算する
    lk = my_loglike(ycal1, obs_data, sigma, n_data, scale=1.0)

    # 返り値に尤度，モル分率，エラー
    return lk,molfraction


# 粒子毎の並列処理
# @profile()
def sim_particle(particle, initial_guess, obs_data, p_pred_bases):
    print('sim_particle')
    # メモリー使用量を参照して，並列化を再initするか決定する
    mem = psutil.virtual_memory() 
    if mem.percent > 80:
        print(mem)
        ray.shutdown()
        time.sleep(10)
        ray.init(num_cpus=n_cores,ignore_reinit_error=True)

    p_pred_bases[:, est_position] = particle
    

    # 並列化計算を実行する
    results = [cal_parallel_new.remote(p_pred_bases[i,:], obs_data, initial_guess) for i in range(n_particle)]

    # 並列計算の結果を取得する
    llk_Cl = ray.get(results)

    # 取得した結果を使える形に成型する
    llk, C_l_ = zip(*llk_Cl)

    return llk,C_l_


# 事前分布の計算？
def cal_prior(theta):
    # σの分を抜く    
    # for i in range(n_state-1):
    # print(theta)
    if normal_pred:
        # 全体のリスト
        all_values = list(range(8))
        # 残りの値
        remaining_values = [x for x in all_values if x not in uni_list]

        if taylor:
            params_p = np.zeros([num_est_params,n_particle])
            for i in remaining_values:
                p = scipy.stats.norm.pdf(theta[:,i],baseparams[i],np.abs(baseparams[i])*coefficent[i])
                # print(p)
                params_p[i,:] = p
                # print(params_p)
            for i in uni_list:
                dl = high_limit[i] - low_limit[i]
                p = scipy.stats.uniform.pdf(theta[:,i], low_limit[i], dl)
                # print(p)
                params_p[i,:] = p
            # print(params_p)
            result = np.prod(params_p.T, axis=1)
            

        else:
            params_p = np.zeros([num_est_params-1,n_particle])
            for i in range(num_est_params-1):
                p = scipy.stats.norm.pdf(theta[:,i],baseparams[i],np.abs(baseparams[i])*coefficent[i])
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


# parity plotを描画，保存する
def ParityplotDrawerWhileSMC(obs_data, dirname01_,dirname02,C, name_):
    tag = ["Xa","Xb","Xc","Xd","Xe","T","u"]
    for i in range(0,5):

        # 箱ひげ図にするためのデータ成型
        Clist = []
        ylist = []
        for k in range(n_data):
            CL = [arr[i][k] for arr in C]
            ylist.append(obs_data[i,k])
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

def DistributionDrawerWhileSMC(p_filt, dirname, name_):
    fig = plt.figure(figsize=(10, 14))
    columns_origin=["Af","Eaf","Ar","Ear","BCO2","dHCO2","BH2O","dHH2O","sigma"]
    for sub in range(p_filt.shape[1]):
        index = sub
        ax1 = fig.add_subplot((num_est_params+2)//3, 3, sub + 1)
        ax1.hist(p_filt[:, sub], 50, range=(low_limit[index], high_limit[index]), density=True)
        ax1.axvline(p_filt[:, sub].mean(),color = 'red', linestyle = 'dashed', linewidth =1)
        # 平均値をグラフのどこかに書き込みたかったが失敗，Excelで簡単に出力できるのでよしとする
        ax1.axvline(baseparams_withsigma[index], color = 'black', linewidth =2)
        ax1.grid('on')
        ax1.set_ylabel(columns_origin[index])
    plt.tight_layout()
    plt.savefig(f"{dirname}{name_}", bbox_inches="tight", dpi=300)
    plt.close()


def SavePosteriorPairplot(p_filt, dirname, name):
    lim = 1
    fig = plt.figure(figsize=(10, 14))
    sub = 0
    columns_origin = ["Af", "Eaf", "Ar", "Ear", "BCO2", "dHCO2", "BH2O", "dHH2O", "sigma"]
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
    plt.savefig(f"{dirname}{name}.png", bbox_inches="tight", dpi=300)
    plt.close()

def SavePosteriorcsv(p_filt, dirname, dirnamepred, name1, name2):
    columns_origin=["Af","Eaf","Ar","Ear","BCO2","dHCO2","BH2O","dHH2O","sigma"]
    columns = []
    for i in range(num_est_params-1):
        columns.append(columns_origin[i])
    columns.append('sigma')
    # df_Posterior = pd.DataFrame(p_filt, columns=["Ha [-]", "Hb [-]", "ka [1/s]", "kb [1/s]", "ba [vol%^-1]", "bb [vol%^-1]"])
    # print(p_filt,columns,p_pred)
    df_Posterior = pd.DataFrame(p_filt, columns = columns)#,"Ar","Ear","BCO2","dHCO2","BH2O","dHH2O"])
    df_Posterior.to_csv(f"{dirname}{name1}.csv", index=False)
    last_pred = f'{dirnamepred}{name2}.csv'
    np.savetxt(last_pred, p_filt, delimiter=',')

def ComparePriorPosterior(firstpred, last_pred, dirname,name):
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
        ax1.axvline(baseparams_withsigma[index], color = 'black', linewidth =2)
        ax1.grid('on')
        ax1.set_ylabel(columns_origin[index])
        sub += 1
    plt.savefig(f"{dirname}{name}.png", bbox_inches="tight", dpi=300)
    plt.close()