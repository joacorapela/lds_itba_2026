
"""
Offline filtering and smoothing with missing observations of a simulated mouse 1D trajectory
============================================================================================

The code below performs offline filtering and smoothing with missing
observations of a simulated mouse trajectory.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
import plotly.graph_objects as go

#%%
# Set initial conditions and parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pos_x0 = 0.0
vel_x0 = 0.0
ace_x0 = 0.0
dt = 1e-3
num_pos = 1000
sigma_a = 1.0
sigma_measurement = 1.0
sqrt_diag_V0_value = 1e-03
h_sec = .1

h_samples = int(round(h_sec / dt))

#%%
# Set LDS parameters
# ~~~~~~~~~~~~~~~~~~

B = np.array([[1, dt, .5*dt**2],
              [0, 1, dt],
              [0, 0, 1]],
             dtype=np.double)
Z = np.array([[1, 0, 0]], dtype=np.double)

Q = sigma_a**2 * np.array([[dt**4/4, dt**3/2, dt**2/2],
                           [dt**3/2, dt**2,   dt],
                           [dt**2/2, dt,      1]],
                          dtype=np.double)
R = np.diag([sigma_measurement**2])
m0 = np.array([pos_x0, vel_x0, ace_x0],
              dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

m0 = np.array([pos_x0, vel_x0, ace_x0],
              dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

#%%
# Sample from the LDS
# ~~~~~~~~~~~~~~~~~~~

# fix random seed
np.random.seed(seed=1234)

M = B.shape[0]
P = Z.shape[0]

# state noise
w = np.random.multivariate_normal(np.zeros(M), Q, num_pos).T
# measurement noise
v = np.random.multivariate_normal(np.zeros(P), R, num_pos).T
# initial state noise
x0 = np.random.multivariate_normal(m0, V0, 1).flatten()
# sample
x = np.empty(shape=(M, num_pos))
y = np.empty(shape=(P, num_pos))
x[:, 0] = B @ x0 + w[:, 0]
for n in range(1, num_pos):
    x[:, n] = B @ x[:, n-1] + w[:, n]
y = Z @ x + v
t = np.arange(num_pos) * dt

y_with_future = y.copy()
missing_start = int(y.shape[1] / 2) - int(h_samples / 2)
missing_slice = slice(missing_start, missing_start + h_samples, 1)
y[0, missing_slice] = np.nan

#%%
# Estimate velocities and accelerations using finite differences
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vel_est_fd = np.diff(y[0, :]) / dt
acc_est_fd = np.diff(vel_est_fd) / dt

#%%
# Perform batch filtering
# ~~~~~~~~~~~~~~~~~~~~~~~

N = y.shape[1]
xnn1 = np.empty(shape=[M, 1, N])
Pnn1 = np.empty(shape=[M, M, N])
xnn = np.empty(shape=[M, 1, N])
Pnn = np.empty(shape=[M, M, N])
innov = np.empty(shape=[P, 1, N])
Sn = np.empty(shape=[P, P, N])
K = np.empty(shape=[M, P, N])

log_like = 0.0
valid_obs_count = 0

for k in range(N):
    if k == 0:
        xnn1[:, 0, 0] = B @ m0
        Pnn1[:, :, 0] = B @ V0 @ B.T + Q
    else:
        xnn1[:, :, k] = B @ xnn[:, :, k-1]
        Pnn1[:, :, k] = B @ Pnn[:, :, k-1] @ B.T + Q
    if np.any(np.isnan(y[:, k])):
        xnn[:, :, k] = xnn1[:, :, k]
        Pnn[:, :, k] = Pnn1[:, :, k]
        innov[:, :, k] = np.nan
        Sn[:, :, k] = np.nan
        K[:, :, k] = np.nan
    else:
        Stmp = Z @ Pnn1[:, :, k] @ Z.T + R
        Sn[:, :, k] = (Stmp + Stmp.T) / 2
        Sinv = np.linalg.inv(Sn[:, :, k])
        K[:, :, k] = Pnn1[:, :, k] @ Z.T @ Sinv
        pred_obs = Z @ xnn1[:, :, k]
        innov[:, 0, k] = y[:, k] - pred_obs[:, 0]
        xnn[:, :, k] = xnn1[:, :, k] + K[:, :, k] @ innov[:, :, k]
        Pnn[:, :, k] = Pnn1[:, :, k] - K[:, :, k] @ Z @ Pnn1[:, :, k]
        log_like -= (np.linalg.slogdet(Sn[:, :, k])[1] +\
                     (innov[:, :, k].T @ Sinv @ innov[:, :, k]).item())
        valid_obs_count += P

log_like = -0.5 * (valid_obs_count * np.log(2 * np.pi) - log_like)
filter_res = {"xnn1": xnn1, "Pnn1": Pnn1, "xnn": xnn, "Pnn": Pnn,
              "innov": innov, "K": K, "Sn": Sn, "log_like": log_like}

#%%
# Perform batch smoothing
# ~~~~~~~~~~~~~~~~~~~~~~~

xnN = np.empty(shape=[M, 1, N])
PnN = np.empty(shape=[M, M, N])
Jn = np.empty(shape=[M, M, N])

xnN[:, :, -1] = xnn[:, :, -1]
PnN[:, :, -1] = Pnn[:, :, -1]
for n in reversed(range(1, N)):
    Jn[:, :, n-1] = Pnn[:, :, n-1] @ B.T @ np.linalg.inv(Pnn1[:, :, n])
    xnN[:, :, n-1] = xnn[:, :, n-1] + \
        Jn[:, :, n-1] @ (xnN[:, :, n]-xnn1[:, :, n])
    PnN[:, :, n-1] = Pnn[:, :, n-1] + \
        Jn[:, :, n-1] @ (PnN[:, :, n]-Pnn1[:, :, n]) @ Jn[:, :, n-1].T
# initial state x00 and V00
# return the smooth estimates of the state at time 0: x0N and V0N
J0 = V0 @ B.T @ np.linalg.inv(Pnn1[:, :, 0])
x0N = np.expand_dims(m0, 1) + J0 @ (xnN[:, :, 0] - xnn1[:, :, 0])
V0N = V0 + J0 @ (PnN[:, :, 0] - Pnn1[:, :, 0]) @ J0.T
smooth_res = {"xnN": xnN, "PnN": PnN, "Jn": Jn, "x0N": x0N, "V0N": V0N,
              "J0": J0}

#%%
# Set variables for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

time = np.arange(0, N*dt, dt)
filtered_means = filter_res["xnn"]
filtered_covs = filter_res["Pnn"]
filtered_std_x = np.sqrt(np.diagonal(a=filtered_covs, axis1=0, axis2=1))
smoothed_means = smooth_res["xnN"]
smoothed_covs = smooth_res["PnN"]
smoothed_std_x = np.sqrt(np.diagonal(a=smoothed_covs, axis1=0, axis2=1))
color_true = "blue"
color_filtered_pattern = "rgba(255,0,0,{:f})"
color_smoothed_pattern = "rgba(0,255,0,{:f})"
cb_alpha = 0.3

#%%
# Define function for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_fig_kinematics_vs_time(
        time,
        true_x,
        other_x,
        filtered_mean_x,
        filtered_ci_x_lower,
        filtered_ci_x_upper,
        smoothed_mean_x,
        smoothed_ci_x_lower,
        smoothed_ci_x_upper,
        cb_alpha,
        color_true,
        color_other,
        color_filtered_pattern,
        color_smoothed_pattern,
        xlabel, ylabel, other_label):

    fig = go.Figure()
    trace_true_x = go.Scatter(
        x=time, y=true_x,
        mode="markers",
        marker={"color": color_true},
        name="true",
        showlegend=True,
    )
    fig.add_trace(trace_true_x)
    if other_x is not None:
        trace_other = go.Scatter(
            x=time, y=other_x,
            mode="markers",
            marker={"color": color_other},
            name=other_label,
            showlegend=True,
        )
        fig.add_trace(trace_other)
    trace_fil_x = go.Scatter(
        x=time, y=filtered_mean_x,
        mode="markers",
        marker={"color": color_filtered_pattern.format(1.0)},
        name="filtered",
        showlegend=True,
        legendgroup="filtered_x",
    )
    fig.add_trace(trace_fil_x)
    trace_fil_x_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([filtered_ci_x_upper,
                          filtered_ci_x_lower[::-1]]),
        fill="toself",
        fillcolor=color_filtered_pattern.format(cb_alpha),
        line=dict(color=color_filtered_pattern.format(0.0)),
        showlegend=False,
        legendgroup="filtered_x",
    )
    fig.add_trace(trace_fil_x_cb)
    trace_smo_x = go.Scatter(
        x=time, y=smoothed_mean_x,
        mode="markers",
        marker={"color": color_smoothed_pattern.format(1.0)},
        name="smoothed",
        showlegend=True,
        legendgroup="smoothed_x",
    )
    fig.add_trace(trace_smo_x)
    trace_smo_x_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([smoothed_ci_x_upper,
                          smoothed_ci_x_lower[::-1]]),
        fill="toself",
        fillcolor=color_smoothed_pattern.format(cb_alpha),
        line=dict(color=color_smoothed_pattern.format(0.0)),
        showlegend=False,
        legendgroup="smoothed_x",
    )
    fig.add_trace(trace_smo_x_cb)

    fig.update_layout(xaxis_title=xlabel,
                      yaxis_title=ylabel,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                     )
    return fig

#%%
# Plot true, measured and smoothed positions (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[0, :]
other_x = y[0, :]
other_label = "measured"
color_other = "black"
filtered_mean_x = filtered_means[0, 0, :]
filtered_ci_x_upper = filtered_mean_x + 1.96*filtered_std_x[:, 0]
filtered_ci_x_lower = filtered_mean_x - 1.96*filtered_std_x[:, 0]
smoothed_mean_x = smoothed_means[0, 0, :]
smoothed_ci_x_upper = smoothed_mean_x + 1.96*smoothed_std_x[:, 0]
smoothed_ci_x_lower = smoothed_mean_x - 1.96*smoothed_std_x[:, 0]

fig = get_fig_kinematics_vs_time(
        time=time,
        true_x=true_x,
        other_x=other_x,
        filtered_mean_x=filtered_mean_x,
        filtered_ci_x_lower=filtered_ci_x_lower,
        filtered_ci_x_upper=filtered_ci_x_upper,
        smoothed_mean_x=smoothed_mean_x,
        smoothed_ci_x_lower=smoothed_ci_x_lower,
        smoothed_ci_x_upper=smoothed_ci_x_upper,
        cb_alpha=cb_alpha,
        color_true=color_true,
        color_other=color_other,
        color_filtered_pattern=color_filtered_pattern,
        color_smoothed_pattern=color_smoothed_pattern,
        xlabel="Time (sec)", ylabel="Position (pixels)",
        other_label=other_label,
)
fig

#%%
# Plot true and smoothed velocities (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[1, :]
other_x = vel_est_fd
other_label = "finite diff"
color_other = "cyan"
filtered_mean_x = filtered_means[1, 0, :]
filtered_ci_x_upper = filtered_mean_x + 1.96*filtered_std_x[:, 1]
filtered_ci_x_lower = filtered_mean_x - 1.96*filtered_std_x[:, 1]
smoothed_mean_x = smoothed_means[1, 0, :]
smoothed_ci_x_upper = smoothed_mean_x + 1.96*smoothed_std_x[:, 1]
smoothed_ci_x_lower = smoothed_mean_x - 1.96*smoothed_std_x[:, 1]

fig = get_fig_kinematics_vs_time(
        time=time,
        true_x=true_x,
        other_x=other_x,
        filtered_mean_x=filtered_mean_x,
        filtered_ci_x_lower=filtered_ci_x_lower,
        filtered_ci_x_upper=filtered_ci_x_upper,
        smoothed_mean_x=smoothed_mean_x,
        smoothed_ci_x_lower=smoothed_ci_x_lower,
        smoothed_ci_x_upper=smoothed_ci_x_upper,
        cb_alpha=cb_alpha,
        color_true=color_true,
        color_other=color_other,
        color_filtered_pattern=color_filtered_pattern,
        color_smoothed_pattern=color_smoothed_pattern,
        xlabel="Time (sec)", ylabel="Velocity (pixels)",
        other_label=other_label,
)

fig

#%%
# Plot true and smoothed accelerations (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[2, :]
other_x = acc_est_fd
other_label = "finite diff"
color_other = "cyan"
filtered_mean_x = filtered_means[2, 0, :]
filtered_ci_x_upper = filtered_mean_x + 1.96*filtered_std_x[:, 2]
filtered_ci_x_lower = filtered_mean_x - 1.96*filtered_std_x[:, 2]
smoothed_mean_x = smoothed_means[2, 0, :]
smoothed_ci_x_upper = smoothed_mean_x + 1.96*smoothed_std_x[:, 2]
smoothed_ci_x_lower = smoothed_mean_x - 1.96*smoothed_std_x[:, 2]

fig = get_fig_kinematics_vs_time(
        time=time,
        true_x=true_x,
        other_x=other_x,
        filtered_mean_x=filtered_mean_x,
        filtered_ci_x_lower=filtered_ci_x_lower,
        filtered_ci_x_upper=filtered_ci_x_upper,
        smoothed_mean_x=smoothed_mean_x,
        smoothed_ci_x_lower=smoothed_ci_x_lower,
        smoothed_ci_x_upper=smoothed_ci_x_upper,
        cb_alpha=cb_alpha,
        color_true=color_true,
        color_other=color_other,
        color_filtered_pattern=color_filtered_pattern,
        color_smoothed_pattern=color_smoothed_pattern,
        xlabel="Time (sec)", ylabel="Accelration (pixels)",
        other_label=other_label,
)

fig

