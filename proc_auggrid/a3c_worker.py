"""
Utils.py
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
        
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        # lp.grad.data.clamp_(-1, 40)
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


"""
Shared adam.py
"""

import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

"""
Envronment:
"""
BTMAX = 0.95
BTMIN = 0.10
C_A = 0.01
C_B = 1.
C_C = 0.
R_MAX = 3.

class Environment:
    def __init__(self, data_env, capacity=50.):
        self.capacity = capacity
        self.data_env = data_env
        self.btmax = capacity*BTMAX
        self.btmin = capacity*BTMIN
        self.pos = 0
        self.state = np.array([self.data_env['GG'].values[0], self.data_env['GC'].values[0], 0., 0., self.btmin, 0., 0])

    def reset(self):
        self.pos = 0
        self.state = np.array([0., 0., 0., 0., self.btmin, 0., 0])
        return self.state
        # return torch.as_tensor(self.state, dtype=torch.float32).squeeze(0)

    def step(self, action):
        charging = 0.0
        discharging = 0.0
        batt_state = 0.0
        grid_state = 0.0
        reward = 0.

        self.state[0] = self.data_env['GG'].values[self.pos]  # PV generation power
        self.state[1] = self.data_env['GC'].values[self.pos]  # Load consumption power
        
        if action == 1: # charge battery
            charging = max(0, self.state[0] - self.state[1])  # Charging power
            charging = min(self.btmax - self.state[4], charging)
            batt_state = charging # Update battery status
            if self.state[0] > self.state[1]:
                reward = 1.
        elif action == 0: # discharge battery
            discharging = min(max(self.state[4]-self.btmin, 0), max(0, self.state[1] - self.state[0])) # discharing power
            batt_state = -discharging # Update battery status
            if self.state[0] < self.state[1]:
                reward = 1.
            
        grid_state = self.state[1] - (self.state[0] - charging + discharging) # Grid power + : import, - : export
        cost = C_A*grid_state**2 + C_B*abs(grid_state) + C_C
        cost = cost if grid_state > 0 else -cost

        normal_v = abs(self.state[0] - self.state[1])
        # lgc = self.state[0] - charging + discharging
        lgc = self.state[1] - max(0, grid_state)
        SC = lgc / self.state[0] if self.state[0] else 0
        SS = lgc / self.state[1] if self.state[1] else 0
        
        # d = abs(grid_state) / normal_v if normal_v else abs(grid_state)
        # reward = min(R_MAX, np.log(1/d)) if d else R_MAX

        # normal_v2 = C_A*normal_v**2 + C_B*abs(normal_v)
        # d = (abs(cost) - C_C) /normal_v2 if normal_v2 else (abs(cost) - C_C)
        # reward = min(R_MAX, np.log(1/d)) if d else R_MAX
                
        self.state[2] = charging
        self.state[3] = discharging
        self.state[4] += batt_state
        self.state[5] = grid_state
        self.state[6] = cost

        done = True if ((self.pos+1) % TS == 0) else False
        # done = True if (reward  > 0) else False
        terminated = False if self.pos+1 < len(self.data_env) else True

        # if done: 
        self.pos += 1
        
        return self.state, reward, done, terminated, self.pos-1


"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import pandas as pd
import os, math

UPDATE_GLOBAL_ITER = 8
GAMMA = 0.999
MAX_EP = 3000

## Load test dataset
TS = 48 # Time steps
local_path = os.getcwd()
if local_path.split('/')[-1] == 'etc_project':
    local_path = os.path.join(local_path, 'proc_auggrid')
df = pd.read_csv(os.path.join(local_path, 'AusGrid_preprocess.csv'), header=[0,1], index_col=0)
df = df.set_index(pd.to_datetime(df.index))
df.columns = df.columns.set_levels(df.columns.levels[0].astype('int64'), level=0)
df = df/1000.
df_date = df.index

customers = sorted(df.columns.levels[0])
data_train = []
samples = list(range(201, 204)) 

for s in samples:
    train = df[s][['GG', 'GC']]
    train['GC'].values[1]
    print(train.shape)
    data_train.append(train)

# env = Environment()
N_S = 7
N_A = 2

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, 64)
        self.pi3 = nn.Linear(64, a_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 64)
        self.v3 = nn.Linear(64, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        # pi1 = self.relu(pi1)
        pi2 = self.pi2(pi1)
        # pi2 = self.relu(pi2)
        logits = self.pi3(pi2)
        # logits = self.tanh(logits)
        
        v1 = torch.tanh(self.v1(x))
        # v1 = self.relu(v1)
        v2 = self.v2(v1)
        # v2 = self.relu(v2)
        values = self.v3(v2)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2) 
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = c_loss.mean() + a_loss.mean()

        return  total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, dataset):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.dataset = dataset
        self.lnet = Net(N_S, N_A)           # local network
        self.env = Environment(dataset, 50.)
        self.lnet.load_state_dict(gnet.state_dict())

    def run(self):
        MAX_EP = len(self.dataset)//TS
        s = self.env.reset()
        
        while True: #ep < MAX_EP:
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.    
            total_step = 1
            terminated = False
            done = False
            
            while done==False:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, terminated, pos = self.env.step(a)

                # print('Action : ', a)
                # if done:
                ep_r += r
            
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                s = s_

                if done or terminated:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done or terminated:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                total_step += 1
            # if ep_r > 0:  
            if terminated:
                break
            
        self.res_queue.put(None)
        
    def run2(self):
        s = self.env.reset()
        MAX_EP = len(self.dataset)
        
        while self.g_ep.value < MAX_EP:
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.    
            total_step = 1
            
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done = self.env.step(a, self.g_ep.value%MAX_EP)

                if ep_r > 6: done = True                                
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


def test(net):
    customers = sorted(df.columns.levels[0])
    data_test = df[1][['GG', 'GC']]
    data_test['GC'].values[1]
    data_test.shape

    df_out = pd.DataFrame(columns=['PV', 'LD', 'PV.C', 'PV.D', 'BT', 'GD', 'COST', 'AC', 'RD'])
    MAX_EP = data_test.shape[0]

    env = Environment(data_test, 50.)
    
    with torch.no_grad():
        s = env.reset()
        for i in range(MAX_EP):
            # logits, _ = net.forward(v_wrap(s[None, :]))
            # prob = F.softmax(logits, dim=1).data
            # a = torch.argmax(prob, dim=1).numpy()[0]
            a = net.choose_action(v_wrap(s[None, :]))
            s_, r, done, _, _ = env.step(a)
            # print('Action : ', a)

            st = np.concatenate((s_, np.array([a-1, r])))
            df_out.loc[i] = st    
            s = s_

    import matplotlib.pyplot as plt

    start_pos = TS*0
    duration = TS*5
    df_dis = df_out[['PV', 'LD', 'BT', 'AC', 'GD', 'RD']]
    df_dis = df_dis.set_index(pd.Index(df_date.values))

    pv =  df_dis['PV'].loc[df_date.values[start_pos]:df_date.values[start_pos+duration]].values
    ld =  df_dis['LD'].loc[df_date.values[start_pos]:df_date.values[start_pos+duration]].values
    bt =  df_dis['BT'].loc[df_date.values[start_pos]:df_date.values[start_pos+duration]].values
    ac =  df_dis['AC'].loc[df_date.values[start_pos]:df_date.values[start_pos+duration]].values

    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df_date.values[start_pos:start_pos+duration+1], pv, label='PV', color='#1f77b4')
    ax1.plot(df_date.values[start_pos:start_pos+duration+1], ld, label='LD', color='#ff7f0e')
    # ax1.plot(df_date.values, ac, label='AC', color='#d62728')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('KW')
    ax1.set_ylim(-1, 4)

    ax2 = ax1.twinx()
    ax2.plot(df_date.values[start_pos:start_pos+duration+1], ac, label='BT', color='#d62728')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Charging/Discharging')
    ax2.set_yticks([])
    ax2.set_ylim(-1, 4)

    print(df_dis.loc[df_date.values[start_pos]:df_date.values[start_pos+duration]])
    plt.show()

        
if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-3, betas=(0.92, 0.999))      # global optimizer
    
    MPATH = os.path.join(local_path, '__pycache__/a3c_gnet.pt')
    if os.path.exists(MPATH):
        gnet.load_state_dict(torch.load(MPATH))


    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, data_train[i]) for i in range(len(data_train))]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    torch.save(gnet.state_dict(), MPATH)

    # gnet.load_state_dict(torch.load(MPATH, weights_only=True))

    # test(gnet)

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
