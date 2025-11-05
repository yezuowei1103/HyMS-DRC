#!/usr/bin/env python
# coding: utf-8
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
# import pandas as pd
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import deque


shift_k = 0

approx_res_size = 1000

model_params = {'tau': 0.0055,
                'nstep': 5000,
                'NIR+minl_6-15': 3,
                'd': 3}

res_params = {'radius': 0.9174228885955388,
              'degree': 227.54718301966668,
              'sigma': 0.9508518064304174,
              'Dr':1000,
             'train_length':25000,
             'NIR+minl_6-15': int(np.floor(approx_res_size/model_params['NIR+minl_6-15']) * model_params['NIR+minl_6-15']),
             'num_inputs': model_params['NIR+minl_6-15'],
             'predict_length': 5000,
             'beta': 0.00001
              }
def generate_reservoir(size, radius, degree, seed=42):
    sparsity = degree / float(size)
    rng = np.random.default_rng(seed)
    A = sparse.rand(size, size, density=sparsity, random_state=rng).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A / e) * radius
    return A
def train(res_params,states,data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['NIR+minl_6-15'])
    states2 = states.copy()
    for j in range(2,np.shape(states2)[0]-2):
        if (np.mod(j,2)==0):
            states2[j,:] = (states[j-1,:]*states[j-2,:]).copy()
    U = np.dot(states2,states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv,np.dot(states2,data.transpose()))
    return Wout.transpose()
def reservoir_layer_with_delay(A, Win, input, res_params, gamma, m):

    N = res_params['NIR+minl_6-15']
    T = res_params['train_length']
    states = np.zeros((N, T))
    state_queue = deque(maxlen=m + 2)
    for _ in range(m + 2):
        state_queue.append(np.zeros(N))
    for i in range(T - 3):
        current_input = np.dot(Win, input[:, i])  # Win * u_t
        internal = np.dot(A, states[:, i])  # Wres * x_t
        delayed_feedback = gamma * state_queue[0]  # γ * x_{t-m}
        # x_{t+1} = tanh(Win*u_t + Wres*x_t + γ*x_{t-m})
        states[:, i + 1] = np.tanh(current_input + internal + delayed_feedback)
        state_queue.append(states[:, i + 1].copy())
    return states, state_queue
def multi_scale_state_fusion_train(scales_params, data_std_T):
    all_states = []
    all_As = []
    all_Wins = []
    all_queues = []
    for i, res_params in enumerate(scales_params):
        gamma = res_params.get('gamma', 0.0)
        m = res_params.get('m', 1)
        A = generate_reservoir(res_params['NIR+minl_6-15'], res_params['radius'], res_params['degree'])
        q = int(res_params['NIR+minl_6-15'] / res_params['num_inputs'])
        Win = np.zeros((res_params['NIR+minl_6-15'], res_params['num_inputs']))
        for j in range(res_params['num_inputs']):
            np.random.seed(seed=j)
            Win[j * q:(j + 1) * q, j] = res_params['sigma'] * (-1 + 2 * np.random.rand(1, q)[0])
        states, state_queue = reservoir_layer_with_delay(
            A, Win, data_std_T[:, :res_params['train_length']], res_params, gamma, m
        )
        all_states.append(states)
        all_As.append(A)
        all_Wins.append(Win)
        all_queues.append(state_queue)
    fused_states = np.vstack(all_states)
    fused_states2 = fused_states.copy()
    for j in range(2, fused_states.shape[0] - 2):
        if j % 2 == 0:
            fused_states2[j, :] = (fused_states[j - 1, :] * fused_states[j - 2, :]).copy()

    beta = scales_params[0]['beta']
    idenmat = beta * sparse.identity(fused_states.shape[0])
    U = np.dot(fused_states2, fused_states2.T) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(fused_states2, data_std_T[:, :scales_params[0]['train_length']].T)).T
    return fused_states[:, -2], Wout, all_As, all_Wins, all_queues, scales_params

def multi_scale_state_fusion_predict(x, Wout, all_As, all_Wins, all_queues, scales_params):
    total_res_size = sum([p['NIR+minl_6-15'] for p in scales_params])
    predict_length = scales_params[0]['predict_length']
    num_inputs = scales_params[0]['num_inputs']
    output = np.zeros((num_inputs, predict_length))

    for t in range(predict_length):
        x_aug = x.copy()
        for j in range(2, total_res_size - 2):
            if j % 3 == 0:
                x_aug[j] = x[j - 1] * x[j - 2]

        out = np.dot(Wout, x_aug)
        out = np.array(out)
        if out.ndim == 2:
            out = out.squeeze()
        if out.ndim > 1:
            out = out.reshape(-1)
        output[:, t] = out
        new_x_parts = []
        idx = 0
        for i, res_params in enumerate(scales_params):
            A = all_As[i]
            Win = all_Wins[i]
            N = res_params['NIR+minl_6-15']
            x_i = x[idx:idx + N]
            state_queue = all_queues[i]
            gamma = res_params.get('gamma', 0.0)
            delayed_feedback = gamma * state_queue[0]
            update = np.dot(Win, out)
            x_next = np.tanh(np.dot(A, x_i) + update + delayed_feedback)
            x_next = np.asarray(x_next).flatten()
            state_queue.append(x_next.copy())
            all_queues[i] = state_queue
            new_x_parts.append(x_next)
            idx += N
        x = np.concatenate(new_x_parts)
    return output.T
data = np.load('lorenz_63.npy')
data = np.transpose(data)
print(data.shape)
model_params = {'tau': 0.005}
train_length = 10000
predict_length = 4500
num_inputs = 3
beta = 1e-5
base_res_size = 200


