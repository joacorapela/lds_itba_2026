
"""
Estimation of DWPA model parameters by grid search
==================================================

The code below estimates parameters of the DWPA model using grid search.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
from tqdm import tqdm
import plotly.graph_objs as go

#%%
# Function to perform Kalman filtering and compute the log-likelihood of LDS model parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def kalman_filter_and_log_like(y, A, Q, C, R, m0, V0):
    M = A.shape[0]
    N = y.shape[1]
    P = C.shape[0]
    xnn1 = np.empty(shape=[M, 1, N])
    Pnn1 = np.empty(shape=[M, M, N])
    xnn = np.empty(shape=[M, 1, N])
    Pnn = np.empty(shape=[M, M, N])
    innov = np.empty(shape=[P, 1, N])
    Sn = np.empty(shape=[P, P, N])
    K = np.empty(shape=[M, P, N])

    log_like = -0.5 * N * P * np.log(2 * np.pi)

    for k in range(N):
        if k == 0:
            xnn1[:, 0, 0] = A @ m0
            Pnn1[:, :, 0] = A @ V0 @ A.T + Q
        else:
            xnn1[:, :, k] = A @ xnn[:, :, k-1]
            Pnn1[:, :, k] = A @ Pnn[:, :, k-1] @ A.T + Q
        Stmp = C @ Pnn1[:, :, k] @ C.T + R
        Sn[:, :, k] = (Stmp + Stmp.T) / 2
        Sinv = np.linalg.inv(Sn[:, :, k])
        K[:, :, k] = Pnn1[:, :, k] @ C.T @ Sinv
        pred_obs = C @ xnn1[:, :, k]
        innov[:, 0, k] = y[:, k] - pred_obs[:, 0]
        xnn[:, :, k] = xnn1[:, :, k] + K[:, :, k] @ innov[:, :, k]
        Pnn[:, :, k] = Pnn1[:, :, k] - K[:, :, k] @ C @ Pnn1[:, :, k]
        log_like -= 0.5 * (np.linalg.slogdet(Sn[:, :, k])[1] +
                           innov[:, :, k].T @ np.linalg.solve(Sn[:, :, k],
                                                              innov[:, :, k]))
    filter_res = {"xnn1": xnn1, "Pnn1": Pnn1, "xnn": xnn, "Pnn": Pnn,
                  "innov": innov, "K": K, "Sn": Sn, "log_like": log_like.item()}
    return filter_res

#%%
# Set hyperparameters
# ~~~~~~~~~~~~~~~~~~~

pos_x0 = 0.0
vel_x0 = 0.0
ace_x0 = 0.0
dt = 1e-3
num_pos = 1000
sigma_a = 1.0
sigma_y = 1.0
sqrt_diag_V0_value = 1e-03

#%%
# Set LDS parameters
# ~~~~~~~~~~~~~~~~~~

A = np.array([[1, dt, .5*dt**2],
              [0, 1, dt],
              [0, 0, 1]],
             dtype=np.double)
C = np.array([[1, 0, 0]], dtype=np.double)

Qe = np.array([[dt**4/4, dt**3/2, dt**2/2],
               [dt**3/2, dt**2,   dt],
               [dt**2/2, dt,      1]],
              dtype=np.double)
Q = sigma_a**2 * Qe
R = np.diag([sigma_y**2])
m0 = np.array([pos_x0, vel_x0, ace_x0],
              dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

#%%
# Sample from the LDS
# ~~~~~~~~~~~~~~~~~~~

# fix random seed
np.random.seed(seed=1234)

M = A.shape[0]
P = C.shape[0]

# state noise
w = np.random.multivariate_normal(np.zeros(M), Q, num_pos).T
# measurement noise
v = np.random.multivariate_normal(np.zeros(P), R, num_pos).T
# initial state noise
x0 = np.random.multivariate_normal(m0, V0, 1).flatten()
# sample
x = np.empty(shape=(M, num_pos))
y = np.empty(shape=(P, num_pos))
x[:, 0] = A @ x0 + w[:, 0]
for n in range(1, num_pos):
    x[:, n] = A @ x[:, n-1] + w[:, n]
y = C @ x + v
t = np.arange(num_pos) * dt

#%%
# Grid search
# ~~~~~~~~~~~

sigma_a_min_seg1 = 0.1
sigma_a_max_seg1 = 2.0
sigma_a_step_seg1 = 0.1
sigma_a_min_seg2 = 3.0
sigma_a_max_seg2 = 40.0
sigma_a_step_seg2 = 1.0
sigma_y_min = 0.9
sigma_y_max = 1.1
sigma_y_step = 0.01
sigma_a_values = np.concatenate([np.arange(sigma_a_min_seg1, sigma_a_max_seg1,
                                           sigma_a_step_seg1),
                                 np.arange(sigma_a_min_seg2, sigma_a_max_seg2,
                                           sigma_a_step_seg2)])
sigma_y_values = np.arange(sigma_y_min, sigma_y_max, sigma_y_step)
ll = np.empty(shape=(len(sigma_a_values), len(sigma_y_values)),
              dtype=np.double)
for i, sigma_a_value in enumerate(
    tqdm(sigma_a_values, desc="Computing LL grid")
):
    Q_gs = Qe * sigma_a_value**2
    for j, sigma_y_value in enumerate(sigma_y_values):
        R_gs = np.diag([sigma_y_value**2])
        ll[i, j] = kalman_filter_and_log_like(y=y, A=A, Q=Q_gs, C=C, R=R_gs,
                                              m0=m0, V0=V0)["log_like"]
ll_argmax_flat = np.argmax(ll)
ll_argmax = np.unravel_index(ll_argmax_flat, ll.shape)
ll_max = ll[ll_argmax]
max_sigma_a = sigma_a_values[ll_argmax[0]]
max_sigma_y = sigma_y_values[ll_argmax[1]]

#%%
# Plot log-likelihoods
# ~~~~~~~~~~~~~~~~~~~~

fig = go.Figure()

# 1. Contour Trace
trace = go.Contour(
    x=sigma_y_values,
    y=sigma_a_values,
    z=ll,
    colorbar=dict(title=dict(text="Log-Likelihood")),
)
fig.add_trace(trace)

# 2. Scatter Trace with Correct Hovertemplate
trace = go.Scatter(
    x=[max_sigma_y],
    y=[max_sigma_a],
    mode="markers",
    marker_color="red",
    marker_size=10,
    marker_symbol="cross",
    hovertemplate=(
        r"$\sigma_y$: %{x:.4f}<br>"
        r"$\sigma_a$: %{y:.4f}<br>"
        r"ll: %{text}<extra></extra>"
    ),
    text=[f"{ll_max:.4f}"],
)
fig.add_trace(trace)

# 3. Layout updates
fig.update_yaxes(title=r"$\sigma_a$", title_font_size=20)
fig.update_xaxes(title=r"$\sigma_y$", title_font_size=20)
fig
