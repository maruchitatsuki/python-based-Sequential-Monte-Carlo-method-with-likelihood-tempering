import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---- model definition ----
def model(t, x, theta):
    k1, k3, k4, k5 = theta
    k2, k6 = 1.0, 0.5  # fixed parameters
    dx1 = -k1 * x[0] + k2 * np.sin(x[1])
    dx2 = -k3 * x[1] + k4 * x[0]**2 - x[2]
    dx3 = -k5 * x[2] + x[1] * x[3]
    dx4 = -k6 * (x[3] - 1) + 0.1 * x[0]
    return [dx1, dx2, dx3, dx4]

# ---- true parameters ----
theta_true = [0.8, 1.2, 0.5, 0.9]

# ---- initial condition, time condition ----
x0 = [1.0, 0.5, 0.2, 1.0]
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 50)

# ---- create true data ----
sol = solve_ivp(model, t_span, x0, args=(theta_true,), t_eval=t_eval)
true_y = np.vstack((sol.y[0], sol.y[2]))  # x1, x3のみ観測対象

# ---- add noise, create artificial data ----
sigma_y = 0.05  # sigma of observation noise
obs_y = true_y + sigma_y * np.random.randn(*true_y.shape)

# ---- save ----
pd.DataFrame(true_y.T, columns=["x1_true", "x3_true"]).to_csv("data_example/true_y.csv", index=False)
pd.DataFrame(obs_y.T, columns=["x1_obs", "x3_obs"]).to_csv("data_example/obs_y.csv", index=False)
pd.DataFrame(t_eval, columns=["time"]).to_csv("data_example/time.csv", index=False)

print("Pseudo data generated and saved: true_y.csv, obs_y.csv, time.csv")

# ---- visualization ----
plt.figure(figsize=(8,5))
plt.plot(t_eval, true_y[0], label="x1 (true)", color="tab:blue")
plt.plot(t_eval, obs_y[0], "o", label="x1 (obs)", color="tab:blue", alpha=0.5)
plt.plot(t_eval, true_y[1], label="x3 (true)", color="tab:orange")
plt.plot(t_eval, obs_y[1], "o", label="x3 (obs)", color="tab:orange", alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("data_example/artificial_data_plot.png", dpi=300)
