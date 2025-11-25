# import
import os
import sys
import time
import json
import psutil
import datetime
import traceback
import requests

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非GUI環境でもプロット可能に
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
import numba
import ray
from memory_profiler import profile

from scipy import integrate, optimize
from assimulo.problem import Implicit_Problem
from assimulo.solvers import Radau5DAE, IDA

from methanation_set_conditon import *
from methanation_set_likelihood import *
from methanation_functions import *
import pylab as P


print(f"numner of particles: {n_particle}, number of cores: {n_cores}, number of data: {n_data}, start time: {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
# 各種保存用のディレクトリの作成
dirname = f"methanation_SMC/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{n_data}/" #Name of the folder (directory) you want to create
os.makedirs(dirname, exist_ok=True)  # Create a folder
dirnamepred = f"{dirname}pred/"
os.makedirs(dirnamepred, exist_ok=True)
dirnameProgress = f"{dirname}tubular_Histgram_Progress/"
os.makedirs(dirnameProgress, exist_ok=True)
dirnameparitybox = f"{dirname}parityplot_boxplot/"
os.makedirs(dirnameparitybox, exist_ok=True)
dirnameparitymean = f"{dirname}parityplot_mean/"
os.makedirs(dirnameparitymean, exist_ok=True)

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
Flow,molfraction = my_model(baseparams, guess)
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

obs_data = data

# Empty array to store the initial distribution
p_pred = np.zeros((n_particle, num_est_params))
# Empty array to store the posterior (filtered) distribution
p_filt = np.zeros((n_particle, num_est_params))
# At the beginning, all particles have equal weights, each assigned 1/n_particle (the total weight sums to 1)
p_weight = np.ones(n_particle) / n_particle
# Used for resampling
p_is = np.zeros(n_particle, dtype=int)
# Data used in likelihood calculations (e.g., outlet concentrations for each experimental condition);
# shape is number of particles × number of data points
y_cal = np.zeros((n_particle, n_data))

d_lk = np.zeros(n_particle)
lk1 = np.zeros(n_particle)
# store base parameters for all particles
p_pred_bases = np.tile(np.append(baseparams, sigma_true), (n_particle, 1))



start_time = time.perf_counter()

# normal_pred= Trueのときは事前分布が正規分布になる
# 事前分布を一様分布として設定
counter = 0
i = 0

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
    # Assign a normal distribution as the prior distribution
    print('normal_distribution')

    # Assign a normal distribution centered on trueparams, with variance defined by Coefficent
    counter = 0
    while counter == num_est_params:
        if num_est_params[i] == 1:
            p_pred[:, i] = np.random.normal(baseparams[i], np.abs(baseparams[i]) * coefficent[i], n_particle)
        counter += 1

    if taylor:
        # Used when assigning a new distribution as the prior distribution 
        # and converting some distributions into uniform ones
        # For example, compare the number of modified parts with the number of estimated parameters,
        # or process cases where the positions of modified parts and the number of elements in uni_list are equal
        print('taylor')
        if len(def_set):
            sys.exit()
            print('There are more Taylor-modified values than estimated parameters. Please check the parameters.')
        else: 
            sys.exit()
            counter_taylor = 0
            j = 0
            while counter_taylor == num_est_params:
                if est_position[j] == uni_list[counter_taylor]:
                    p_pred[:, counter] = np.random.uniform(low_limit[counter_taylor], high_limit[counter_taylor], n_particle)
                    j += 1
                counter_taylor += 1

firstpred = f'{dirnamepred}first_p_pred.csv'
np.savetxt(firstpred, p_pred, delimiter=',')

# Plot the prior distribution
# This histogram is not yet wrapped in a function, but it could be made into one
DistributionDrawerWhileSMC(p_pred, dirnameProgress, "00_PriorDistribution")
# Initial parallelization (parallel expansion)
ray.init(num_cpus=n_cores, ignore_reinit_error=True)
RAY_DEDUP_LOGS = 0


try:

    # 初期分布の状態で尤度推定＆parity plot用のデータ取得
    lk,C_l_= sim_particle(p_pred,guess, obs_data, p_pred_bases)
    print('calculation of initial likelihood done')
    time1 = time.time()

    # parity plotを描く
    ParityplotDrawerWhileSMC(obs_data, dirnameparitybox,dirnameparitymean, C_l_, f"00_PriorDistribution")

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
                    lk2,C_l_ = sim_particle(p_pred, guess, obs_data, p_pred_bases)
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

                    lk2,C_l_ = sim_particle(p_pred, guess, obs_data, p_pred_bases)
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
                lk2,C_l_ = sim_particle(p_pred, guess, obs_data, p_pred_bases)
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

        ParityplotDrawerWhileSMC(obs_data, dirnameparitybox,dirnameparitymean, C_l_,f"step=0{step}_nMH=0{j}")
        np.savetxt(f'{dirnamepred}{step}_p_pred.csv', p_pred, delimiter=',')
        DistributionDrawerWhileSMC(p_filt, dirnameProgress, f"SMC_Histgram_Progress_step=0{step}_nMH=0{j}")


    if gamma_new < 1.0:
        print("tempering does't complete: last gamma =", gamma_new)

    end_time = time.time() - start_time
    print(f"end_time:{end_time}")

    DistributionDrawerWhileSMC(p_filt, dirname, f"SMC_Posterior_Distribution")
    SavePosteriorPairplot(p_filt, dirname, "Posterior_Pairplot")
    name1 = "Posterior_Distribution"
    name2 = "last_p_pred"
    SavePosteriorcsv(p_filt, dirname, dirnamepred, name1, name2)
    last_pred = f'{dirnamepred}{name2}.csv'
    ComparePriorPosterior(firstpred, last_pred, dirname, "Histgram_compare")

except Exception as e:
    # 発生中の例外に関する情報を取得する
    etype, value, tb = sys.exc_info()
    
    traceback.format_exception(etype, value, tb) # list[str] を返す
    traceback.format_tb(tb) # list[str] を返す
    traceback.format_exception_only(etype, value) # list[str] を返す
    traceback.format_exc() # tuple[str] を返す
    print(traceback.format_exception(etype, value, tb))
    print(f"{e.__class__.__name__}: {e}")
