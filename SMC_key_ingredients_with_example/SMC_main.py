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

from set_condition import *
import pylab as P


print('start')

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
# Array to store some likelihood-related values (one element per particle)
d_lk = np.zeros(n_particle)
# Array to store some likelihood-related values (one element per particle)
lk1 = np.zeros(n_particle)



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
    requests.post(WEB_HOOK_URL,data=json.dumps({
    "text" : f"エラー発生,{e.__class__.__name__}: {e}"
    }))
