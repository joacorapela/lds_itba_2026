
"""
Sampling from a 1D Discrete Wiener Process Acceleration Model
=============================================================

The code below samples from a 1D Discrete Wiener Process Acceleration (DWPA)
model.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
import plotly.graph_objs as go


#%%
# Set hyperparameters
# ~~~~~~~~~~~~~~~~~~~

pos_x0 = 0.0
vel_x0 = 0.0
ace_x0 = 0.0
dt = 1e-3
num_pos = 1000
sigma_a = 1.0
sigma_measurement = 1.0
sqrt_diag_V0_value = 1e-03

#%%
# Set LDS parameters
# ~~~~~~~~~~~~~~~~~~

A = np.array([[1, dt, .5*dt**2],
              [0, 1, dt],
              [0, 0, 1]],
             dtype=np.double)
C = np.array([[1, 0, 0]], dtype=np.double)

Q = sigma_a**2 * np.array([[dt**4/4, dt**3/2, dt**2/2],
                           [dt**3/2, dt**2,   dt],
                           [dt**2/2, dt,      1]],
                          dtype=np.double)
R = np.diag([sigma_measurement**2])
m0 = np.array([pos_x0, vel_x0, ace_x0],
              dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

#%%
# Sample from the LDS
# ~~~~~~~~~~~~~~~~~~~

# fix random seed
np.random.seed(seed=1234)

M = A.shape[0]
N = C.shape[0]

# state noise
w = np.random.multivariate_normal(np.zeros(M), Q, num_pos).T
# measurement noise
v = np.random.multivariate_normal(np.zeros(N), R, num_pos).T
# initial state noise
x0 = np.random.multivariate_normal(m0, V0, 1).flatten()
# sample
x = np.empty(shape=(M, num_pos))
y = np.empty(shape=(N, num_pos))
x[:, 0] = A @ x0 + w[:, 0]
for n in range(1, num_pos):
    x[:, n] = A @ x[:, n-1] + w[:, n]
y = C @ x + v
t = np.arange(num_pos) * dt

#%%
# Plot state positions and measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig = go.Figure()
trace = go.Scatter(x=t, y=x[0, :], mode="markers",
                   showlegend=True, name="state position")
fig.add_trace(trace)
trace = go.Scatter(x=t, y=x[1, :], mode="markers",
                   showlegend=True, name="state velocity")
fig.add_trace(trace)
trace = go.Scatter(x=t, y=x[2, :], mode="markers",
                   showlegend=True, name="state acceleration")
fig.add_trace(trace)
trace = go.Scatter(x=[t[0]], y=[x0[0]], mode="markers",
                   text="initial state position", marker={"size": 7},
                   showlegend=False)
fig.add_trace(trace)
trace = go.Scatter(x=t, y=y[0, :], mode="markers",
                   showlegend=True, name="measured position")
fig.add_trace(trace)
fig.update_xaxes(title="Time (sec)")
fig
