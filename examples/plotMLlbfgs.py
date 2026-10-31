
"""
Estimation of DWPA model parameters by grid search and gradient-based optimisation
==================================================================================

The code below estimates parameters of the DWPA model using grid search and a
quasi-Newton optimisation method, LBFGS.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import time
import math
import numpy as np
import torch
from tqdm import tqdm
import plotly.graph_objs as go

torch.autograd.set_detect_anomaly(True)

#%%
# Function to compute the log-likelihood of LDS model parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_log_like(y, A, Q, C, R, m0, V0):
    N = y.shape[1]
    P = C.shape[0]

    # Initial log-likelihood scalar/tensor
    log_like = torch.tensor(
        -0.5 * N * P * math.log(2 * math.pi),
        dtype=torch.double,
    )

    for n in range(N):
        if n == 0:
            # Out-of-place assignments for initial step
            xnn1 = A @ torch.unsqueeze(m0, -1)
            Pnn1 = A @ V0 @ A.T + Q
        else:
            xnn1 = A @ xnn
            Pnn1 = A @ Pnn @ A.T + Q

        Stmp = C @ Pnn1 @ C.T + R
        Sn = (Stmp + Stmp.T) / 2

        # Solve for inverse cleanly
        Sinv = torch.linalg.solve(
            Sn, torch.eye(P, dtype=torch.double),
        )

        K = Pnn1 @ C.T @ Sinv

        pred_obs = C @ xnn1

        # Handle observation slice dimensions safely
        yn = y[:, n:(n + 1)]  # keeps shape [P, 1]
        innov = yn - pred_obs

        xnn = xnn1 + K @ innov
        Pnn = Pnn1 - K @ C @ Pnn1

        # Out-of-place log-likelihood update
        log_like = log_like - 0.5 * (
            torch.logdet(Sn) + innov.T @ torch.linalg.solve(Sn, innov)
        )

    return log_like

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

A = torch.tensor([[1, dt, dt**2/2],
                  [0, 1, dt],
                  [0, 0, 1]],
                 dtype=torch.double)
C = torch.tensor([[1, 0, 0]], dtype=torch.double)

Qe = torch.tensor([[dt**4/4, dt**3/2, dt**2/2],
                   [dt**3/2, dt**2,   dt],
                   [dt**2/2, dt,      1]],
                  dtype=torch.double)
Q = sigma_a**2 * Qe
R = torch.diag(torch.tensor([sigma_y**2]))
m0 = torch.tensor([pos_x0, vel_x0, ace_x0],
                 dtype=torch.double)
V0 = torch.diag(torch.ones_like(m0)*sqrt_diag_V0_value**2)

#%%
# Sample from the LDS
# ~~~~~~~~~~~~~~~~~~~

start_time = time.time()

# fix random seed
np.random.seed(seed=1234)

M = A.shape[0]
P = C.shape[0]

# state noise
w = np.random.multivariate_normal(np.zeros(M), Q.numpy(), num_pos).T
# measurement noise
v = np.random.multivariate_normal(np.zeros(P), R.numpy(), num_pos).T
# initial state noise
x0 = np.random.multivariate_normal(m0.numpy(), V0.numpy(), 1).flatten()
# sample
x = np.empty(shape=(M, num_pos))
y = np.empty(shape=(P, num_pos))
x[:, 0] = A.numpy() @ x0 + w[:, 0]
for n in range(1, num_pos):
    x[:, n] = A.numpy() @ x[:, n-1] + w[:, n]
y = C.numpy() @ x + v
t = np.arange(num_pos) * dt

elapsed_time = time.time() - start_time

y = torch.from_numpy(y)
t = torch.from_numpy(t)

#%%
# Build grid of log-likelihood values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sigma_a_min_seg1 = 0.1
sigma_a_max_seg1 = 2.0
sigma_a_step_seg1 = 0.1
sigma_a_min_seg2 = 3.0
sigma_a_max_seg2 = 40.0
sigma_a_step_seg2 = 1.0
sigma_y_min = 0.9
sigma_y_max = 1.1
sigma_y_step = 0.01
sigma_a_values = torch.concatenate([torch.arange(sigma_a_min_seg1, sigma_a_max_seg1,
                                                 sigma_a_step_seg1),
                                    torch.arange(sigma_a_min_seg2, sigma_a_max_seg2,
                                                 sigma_a_step_seg2)])
sigma_y_values = torch.arange(sigma_y_min, sigma_y_max, sigma_y_step)
ll_gs = torch.empty(size=(len(sigma_a_values), len(sigma_y_values)),
                dtype=torch.double)
for i, sigma_a_value in enumerate(
    tqdm(sigma_a_values, desc="Computing LL grid")
):
    Q_gs = Qe * sigma_a_value**2
    for j, sigma_y_value in enumerate(sigma_y_values):
        R_gs = torch.diag(torch.tensor([sigma_y_value**2], dtype=torch.double))
        ll_gs[i, j] = compute_log_like(y=y, A=A, Q=Q_gs, C=C, R=R_gs, m0=m0,
                                       V0=V0)
#%%
# Find grid maximum values
# ~~~~~~~~~~~~~~~~~~~~~~~~

ll_gs_argmax_flat = torch.argmax(ll_gs)
ll_gs_argmax = torch.unravel_index(ll_gs_argmax_flat, ll_gs.shape)
ll_gs_max = ll_gs[ll_gs_argmax]
max_sigma_a_gs = sigma_a_values[ll_gs_argmax[0]]
max_sigma_y_gs = sigma_y_values[ll_gs_argmax[1]]

#%%
# Run a quasi-Newton (LBFGS) maximisation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define variables that control the optimisation

max_iter = 30
lr = 1.0
tolerance_grad = 1e-6
tolerance_change = 1e-6
line_search_fn = "strong_wolfe"

# Set initial conditions
sigma_a0 = 39.0
sigma_y0 = 1.1

# perform optimisation
optim_params = {
    "max_iter": 1,
    "lr": lr,
    "tolerance_grad": tolerance_grad,
    "tolerance_change": tolerance_change,
    "line_search_fn": line_search_fn,
}
sigma_a_tensor = torch.tensor([sigma_a0], dtype=torch.double)
sigma_y_tensor = torch.tensor([sigma_y0], dtype=torch.double)
sigma_a_tensor.requires_grad = True
sigma_y_tensor.requires_grad = True
optimizer = torch.optim.LBFGS([sigma_a_tensor, sigma_y_tensor], **optim_params)


def closure():
    optimizer.zero_grad()

    R_lbfgs = (sigma_y_tensor**2).reshape(P, P)
    Q_lbfgs = sigma_a_tensor**2 * Qe
    log_like = compute_log_like(
        y=y, A=A, Q=Q_lbfgs, C=C, R=R_lbfgs, m0=m0, V0=V0
    )
    loss = -log_like
    loss.backward(retain_graph=True)
    return loss


# Run LBFGS optimisation
print(f"{'Iteration':<10} | {'Loss (Neg Log-Likelihood)':<25} | {'sigma_a':<12} | {'sigma_y':<12}")
print("-" * 65)

R_lbfgs = (sigma_y_tensor**2).reshape(P, P)
Q_lbfgs = sigma_a_tensor**2 * Qe
ll0 = compute_log_like(
    y=y, A=A, Q=Q_lbfgs, C=C, R=R_lbfgs, m0=m0, V0=V0
)

hist_ll = [ll0]
hist_sigma_a = [sigma_a0]
hist_sigma_y = [sigma_y0]
for epoch in range(max_iter):
    # Perform optimization step
    optimizer.step(closure)

    # Save history of parameters and log-likelihoods
    R_lbfgs = (sigma_y_tensor**2).reshape(P, P)
    Q_lbfgs = sigma_a_tensor**2 * Qe
    lln = compute_log_like(
        y=y, A=A, Q=Q_lbfgs, C=C, R=R_lbfgs, m0=m0, V0=V0
    )

    hist_ll.append(lln)
    hist_sigma_a.append(sigma_a_tensor.item())
    hist_sigma_y.append(sigma_y_tensor.item())

    # Print progress metrics
    print(f"{epoch+1:<10} | {lln.item():<25.4f} |"
          f" {sigma_a_tensor.item():<12.4f} | {sigma_y_tensor.item():<12.4f}")

#%%
# Plot log-likelihoods
# ~~~~~~~~~~~~~~~~~~~~

fig = go.Figure()

# Contour plot of log-likelihoods
trace = go.Contour(
    x=sigma_y_values,
    y=sigma_a_values,
    z=ll_gs,
    colorbar=dict(title=dict(text="Log-Likelihood")),
)
fig.add_trace(trace)

# Grid search maximum cross
trace = go.Scatter(
    x=[max_sigma_y_gs],
    y=[max_sigma_a_gs],
    mode="markers",
    marker_color="red",
    marker_size=10,
    marker_symbol="cross",
    hovertemplate=(
        r"$\sigma_ys$: %{x:.4f}<br>"
        r"$\sigma_a$: %{y:.4f}<br>"
        r"ll: %{text}<extra></extra>"
    ),
    text=[f"{ll_gs_max:.4f}"],
    showlegend=False,
)
fig.add_trace(trace)

# LBFGS optimisation path
trace = go.Scatter(
    x=hist_sigma_y,
    y=hist_sigma_a,
    mode="lines+markers",
    marker_color="green",
    hovertemplate=(
        r"$\sigma_y$: %{x:.4f}<br>"
        r"$\sigma_a$: %{y:.4f}<br>"
        r"%{text}<extra></extra>"
    ),
    text=[f"ll: {ll_item.item():.4f}" for ll_item in hist_ll],
    showlegend=False,
)
fig.add_trace(trace)

fig.update_yaxes(title=r"$\sigma_a$", title_font_size=20)
fig.update_xaxes(title=r"$\sigma_y$", title_font_size=20)
fig
