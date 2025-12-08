# mm_generate_data.py

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

# Michaelis-Menten ODE: dS/dt
def mm_ode(t, S, Vmax, Km):
    return - Vmax * S / (Km + S)

def simulate_mm(Vmax, Km, S0, t_span, num_points=50):
    """
    Michaelis-Menten 進行曲線（基質濃度）を数値的に解く。
    戻り値: t, S(t), P(t) = S0 - S(t)
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    sol = solve_ivp(
        fun=lambda t, S: mm_ode(t, S, Vmax, Km),
        t_span=t_span,
        y0=[S0],
        t_eval=t_eval,
        method="RK45",
    )

    t = sol.t
    S = sol.y[0]
    P = S0 - S
    return t, S, P

def make_pseudo_data(
    Vmax_true=1.2,
    Km_true=0.5,
    S0_list = [0.1, 0.25, 0.5, 1.0, 2.0],
    t_span=(0.0, 10.0),
    num_points=40,
    noise_std=0.02,
    csv_path="mm_pseudo_data",
):
    """
    真のパラメータから疑似データを生成し、CSVに保存する。
    保存する列:
      t, S_true, P_true, P_obs
    """
    i = 0
    for S0 in S0_list:
        i += 1
        seed=20250205 + i
        np.random.seed(seed)

        t, S_true, P_true = simulate_mm(Vmax_true, Km_true, S0, t_span, num_points)

        # ガウスノイズ付き観測（生成物濃度）
        P_obs = P_true + np.random.normal(0.0, noise_std, size=len(P_true))

        df = pd.DataFrame(
            {
                "t": t,
                "S_true": S_true,
                "P_true": P_true,
                "P_obs": P_obs,
            }
        )
        df.to_csv(f"data/{csv_path}_{i}.csv", index=False)
        print(f"Saved pseudo data to: {csv_path}")
        print(f"True parameters: Vmax={Vmax_true}, Km={Km_true}, S0={S0}, noise_std={noise_std}")

if __name__ == "__main__":
    # 必要に応じてここを書き換えればOK
    make_pseudo_data()
