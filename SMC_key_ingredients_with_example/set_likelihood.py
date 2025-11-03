import numpy as np
from scipy.integrate import solve_ivp


def my_model(t, x, theta):
    k1, k3, k4, k5 = theta
    k2, k6 = 1.0, 0.5  # 固定パラメータ
    dx1 = -k1 * x[0] + k2 * np.sin(x[1])
    dx2 = -k3 * x[1] + k4 * x[0]**2 - x[2]
    dx3 = -k5 * x[2] + x[1] * x[3]
    dx4 = -k6 * (x[3] - 1) + 0.1 * x[0]
    return [dx1, dx2, dx3, dx4]

def simulate(theta, t_span, x0):
    sol = solve_ivp(model, t_span, x0, args=(theta,), t_eval=np.linspace(t_span[0], t_span[1], 50))
    return sol.t, sol.y


def log_likelihood(theta, y_obs, t_obs, x0, sigma_y):
    t_sim, y_sim = simulate(theta, [t_obs[0], t_obs[-1]], x0)
    y_model = np.vstack((y_sim[0], y_sim[2]))  # x1とx3のみ観測
    # 補間して同じ時間点へ
    y_model_interp = np.array([np.interp(t_obs, t_sim, y_model[i]) for i in range(2)])
    residual = y_obs - y_model_interp
    N, m = y_obs.shape[1], y_obs.shape[0]
    ll = -0.5 * N * m * np.log(2*np.pi*sigma_y**2) - 0.5 * np.sum(residual**2) / sigma_y**2
    return ll