# import
import os
import sys
import time
import json
import psutil
import datetime
import traceback
import requests
import scipy

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

from Micmem_settings import *
from Micmem_likelihood import *
import pylab as P


print(f"numner of particles: {n_particle}, number of cores: {n_cores}, number of data: {n_ex}, start time: {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
# 各種保存用のディレクトリの作成
# dirname = f"methanation_SMC/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{n_ex}/" #Name of the folder (directory) you want to create
# os.makedirs(dirname, exist_ok=True)  # Create a folder
# dirnamepred = f"{dirname}pred/"
# os.makedirs(dirnamepred, exist_ok=True)
# dirnameProgress = f"{dirname}tubular_Histgram_Progress/"
# os.makedirs(dirnameProgress, exist_ok=True)
# dirnameparitybox = f"{dirname}parityplot_boxplot/"
# os.makedirs(dirnameparitybox, exist_ok=True)
# dirnameparitymean = f"{dirname}parityplot_mean/"
# os.makedirs(dirnameparitymean, exist_ok=True)



start_time = time.perf_counter()

# np.savetxt(f'{dirnamepred}first_p_pred.csv', first_p_pred, delimiter=',')

# Plot the prior distribution
# This histogram is not yet wrapped in a function, but it could be made into one
# DistributionDrawerWhileSMC(p_pred, dirnameProgress, "00_PriorDistribution")
# Initial parallelization (parallel expansion)
ray.init(num_cpus=n_cores, ignore_reinit_error=True)
RAY_DEDUP_LOGS = 0


def cal_prior(theta, priors):
    """
    theta: shape = (n_particle, num_params)
    priors: dict (PyMC風)
    """
    n_particle = theta.shape[0]
    num_params = len(priors)

    # param-by-param pdf
    pdf_vals = np.zeros((n_particle, num_params))

    for j, (name, cfg) in enumerate(priors.items()):
        x = theta[:, j]

        if cfg["dist"] == "normal":
            mu = cfg["mu"]
            sigma = cfg["sigma"]
            pdf_vals[:, j] = scipy.stats.norm.pdf(x, loc=mu, scale=sigma)

        elif cfg["dist"] == "uniform":
            low = cfg["low"]
            high = cfg["high"]
            pdf_vals[:, j] = scipy.stats.uniform.pdf(x, loc=low, scale=high - low)

        else:
            raise ValueError(f"Unknown prior: {cfg['dist']}")

    # 全パラメータの独立 pdf の積 → prior(theta)
    prior_total = np.prod(pdf_vals, axis=1)

    return prior_total




try:

    # 初期分布の状態で尤度推定＆parity plot用のデータ取得
    lk,C_l_= sim_particle(p_pred)
    print('calculation of initial likelihood done')
    time1 = time.time()

    # parity plotを描く
    # ParityplotDrawerWhileSMC(obs_data, dirnameparitybox,dirnameparitymean, C_l_, f"00_PriorDistribution")

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



            p0_1 = cal_prior(p_filt, priors)
            p0_2 = cal_prior(p_pred, priors)
            p0 = np.int32(p0_2 > 0)

            p_pred = p_pred * p0[:, None] + p_filt * (1.0 - p0[:, None])
            lk2,C_l_ = sim_particle(p_pred)

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

        # ParityplotDrawerWhileSMC(obs_data, dirnameparitybox,dirnameparitymean, C_l_,f"step=0{step}_nMH=0{j}")
        # np.savetxt(f'{dirnamepred}{step}_p_pred.csv', p_pred, delimiter=',')
        # DistributionDrawerWhileSMC(p_filt, dirnameProgress, f"SMC_Histgram_Progress_step=0{step}_nMH=0{j}")


    if gamma_new < 1.0:
        print("tempering does't complete: last gamma =", gamma_new)

    end_time = time.time() - start_time
    print(f"end_time:{end_time}")

    # print(p_pred)
    Vmax  = p_pred[:, 0]
    Km    = p_pred[:, 1]
    sigma = p_pred[:, 2]
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.hist(Vmax, bins=20, color='skyblue', edgecolor='black')
    plt.title("Vmax distribution")

    plt.subplot(1,3,2)
    plt.hist(Km, bins=20, color='salmon', edgecolor='black')
    plt.title("Km distribution")

    plt.subplot(1,3,3)
    plt.hist(sigma, bins=20, color='lightgreen', edgecolor='black')
    plt.title("sigma distribution")

    plt.tight_layout()
    plt.savefig(f"Posterior_Distributions.png", dpi=300)

    # DistributionDrawerWhileSMC(p_filt, dirname, f"SMC_Posterior_Distribution")
    # SavePosteriorPairplot(p_filt, dirname, "Posterior_Pairplot")
    # name1 = "Posterior_Distribution"
    # name2 = "last_p_pred"
    # SavePosteriorcsv(p_filt, dirname, dirnamepred, name1, name2)
    # last_pred = f'{dirnamepred}{name2}.csv'
    # ComparePriorPosterior(firstpred, last_pred, dirname, "Histgram_compare")

except Exception as e:
    # 発生中の例外に関する情報を取得する
    etype, value, tb = sys.exc_info()
    
    traceback.format_exception(etype, value, tb) # list[str] を返す
    traceback.format_tb(tb) # list[str] を返す
    traceback.format_exception_only(etype, value) # list[str] を返す
    traceback.format_exc() # tuple[str] を返す
    print(traceback.format_exception(etype, value, tb))
    print(f"{e.__class__.__name__}: {e}")
