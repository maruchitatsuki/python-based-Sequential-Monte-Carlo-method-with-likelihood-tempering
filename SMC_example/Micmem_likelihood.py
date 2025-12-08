import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy
import psutil
import ray
# === Assimulo (DAE Solvers) ===
from assimulo.solvers import Radau5DAE, IDA
from assimulo.problem import Implicit_Problem

from Micmem_settings import *

# --- ODEとシミュレーション（t配列を指定する版） ---
def mm_ode(t, S, Vmax, Km):
    return - Vmax * S / (Km + S)

def simulate_mm_on_grid(Vmax, Km, S0, t_array):
    """
    観測時刻 t_array に合わせてミカエリスメンテン進行曲線を解く。
    戻り値: P_model(t_array)
    """
    t_span = (t_array[0], t_array[-1])

    sol = solve_ivp(
        fun=lambda t, S: mm_ode(t, S, Vmax, Km),
        t_span=t_span,
        y0=[S0],
        t_eval=t_array,
        method="RK45",
    )
    S_model = sol.y[0]
    P_model = S0 - S_model
    return P_model

@ray.remote
def log_likelihood_mm_multi(params):
    """
    複数の S0 条件のデータを使った対数尤度関数。
    
    params  : [Vmax, Km, sigma]
    dataset : load_all_mm_data() で読み込んだデータのリスト
    """
        
    Vmax, Km, sigma = params

    # σを推定しない場合，sigmaはsigma_trueに固定される
    if est_sigma:
        sigma = params[-1]
    else:
        sigma = sigma_true


    if sigma <= 0:
        return -np.inf

    logL_total = 0.0
    P_model_list = []

    for i in range(n_ex):
        t     = dataset[i]["t"]
        P_obs = dataset[i]["P_obs"]
        S0    = dataset[i]["S0"]

        # モデル予測
        P_model = simulate_mm_on_grid(Vmax, Km, S0, t)

        # 残差
        residual = P_obs - P_model

        logL_i = -0.5 * datapoint * np.log(2*np.pi*sigma**2) \
                 - np.sum(residual**2) / (2*sigma**2)

        logL_total += logL_i
        P_model_list.append(P_model)


    return logL_total, P_model_list

def sim_particle(particle):
    print('sim_particle')

    # 並列化計算を実行する
    results = [log_likelihood_mm_multi.remote(particle[i,:]) for i in range(n_particle)]
    # print(results)

    # 並列計算の結果を取得する
    llk_Cl = ray.get(results)

    # 取得した結果を使える形に成型する
    llk, C_l_ = zip(*llk_Cl)

    return llk,C_l_


if __name__ == "__main__":
    dataset = []
    n_cond = 6
    base_path="data/mm_pseudo_data"
    for i in range(0, n_cond):
        df = pd.read_csv(f"{base_path}_{i}.csv")
        data_i = {
            "t": df["t"].values,
            "P_obs": df["P_obs"].values,
            "S0": df["S_true"].iloc[0],
        }
        dataset.append(data_i)
    params = [1.0, 0.4, 0.05]  # テスト用パラメータ
    datapoint = len(dataset[0]["t"])
    log_lk = log_likelihood_mm_multi(params, dataset, n_cond, datapoint)
    print('yeee')
    print(f"Log-likelihood: {log_lk}")