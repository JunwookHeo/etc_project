from torch import nn
import torch
import numpy as np

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
        return torch.as_tensor(self.state, dtype=torch.float32).squeeze(0)

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
            # if self.state[0] > self.state[1]:
            #     reward = 1.0
        elif action == 0: # discharge battery
            discharging = min(max(self.state[4]-self.btmin, 0), max(0, self.state[1] - self.state[0])) # discharing power
            batt_state = -discharging # Update battery status
            # if self.state[0] < self.state[1]:
            #     reward = 1.
            
        grid_state = self.state[1] - (self.state[0] - charging + discharging) # Grid power + : import, - : export
        cost = C_A*grid_state**2 + C_B*abs(grid_state) + C_C
        cost = cost if grid_state > 0 else -cost

        normal_v = abs(self.state[0] - self.state[1])       
        d = abs(grid_state) / normal_v if normal_v else abs(grid_state)
        reward = max((min(R_MAX, np.log(1/d)) if d else R_MAX), 0)

        # normal_v2 = C_A*normal_v**2 + C_B*abs(normal_v)
        # d = (abs(cost) - C_C) /normal_v2 if normal_v2 else (abs(cost) - C_C)
        # reward = min(R_MAX, np.log(1/d)) if d else R_MAX
                
        self.state[2] = charging
        self.state[3] = discharging
        self.state[4] += batt_state
        self.state[5] = grid_state
        self.state[6] = cost

        done = ((self.pos+1) % TS == 0)
        self.pos += 1
        
        return torch.as_tensor(self.state, dtype=torch.float32), torch.tensor([reward]), done 


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
GAMMA = 0.800
MAX_EP = 3000

## Load test dataset
TS = 48 # Time steps

def load_train_data(local_path):    
    df = pd.read_csv(os.path.join(local_path, 'AusGrid_preprocess.csv'), header=[0,1], index_col=0)
    df = df.set_index(pd.to_datetime(df.index))
    df.columns = df.columns.set_levels(df.columns.levels[0].astype('int64'), level=0)
    df = df/1000.
    df_date = df.index

    customers = sorted(df.columns.levels[0])
    data_train = []
    samples = list(range(201, 211)) 

    for s in samples:
        train = df[s][['GG', 'GC']]
        train['GC'].values[1]
        print('samples', train.shape)
        data_train.append(train)

    return data_train

def load_test_data(local_path):
    df = pd.read_csv(os.path.join(local_path, 'AusGrid_preprocess.csv'), header=[0,1], index_col=0)
    df = df.set_index(pd.to_datetime(df.index))
    df.columns = df.columns.set_levels(df.columns.levels[0].astype('int64'), level=0)
    df = df/1000.
    df_date = df.index
    
    customers = sorted(df.columns.levels[0])
    data_test = df[1][['GG', 'GC']]
    # data_test['GC'].values[1]
    # data_test.shape
    return data_test, df_date
    
# env = Environment()
N_S = 7
N_A = 2

class a3c(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(a3c, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 256)
        self.pi2 = nn.Linear(256, 128)
        self.pi3 = nn.Linear(128, a_dim)
        
        self.v1 = nn.Linear(s_dim, 256)
        self.v2 = nn.Linear(256, 128)
        self.v3 = nn.Linear(128, 1)
        # self.distribution = torch.distributions.Categorical

    def forward(self, x):
        px = F.relu(self.pi1(x))
        px = F.relu(self.pi2(px))
        logits = F.softmax(self.pi3(px), dim=-1)
                
        vx = F.relu(self.v1(x))        
        vx = F.relu(self.v2(vx))        
        values = self.v3(vx)
        return values, logits

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, dataset):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.dataset = dataset
        self.lnet = a3c(N_S, N_A)           # local network
        self.env = Environment(dataset, 50.)
        self.lnet.load_state_dict(gnet.state_dict())

    def run(self):
        MAX_EP = len(self.dataset)
        state = self.env.reset()

        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0
        
        log_probs = []
        values = []
        rewards = []

        total_export = 0.0
        total_pv = 0.0
        total_ld = 0.0
        
        self.lnet.load_state_dict(self.gnet.state_dict())
        for i in range(MAX_EP):
            value, logits = self.lnet.forward(state)
            value = value.detach().numpy()[0]
            dist = logits.detach().numpy()
            action = np.random.choice(N_A, p=np.squeeze(dist))
        
            log_prob = torch.log(logits.squeeze(0)[action])
            entropy = -np.sum(dist * np.log(dist + 1e-10))
            new_state, reward, done = self.env.step(action)
        
            rewards.append(reward.numpy())
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            total_export += state.numpy()[5] if state.numpy()[5] < 0 else 0
            total_pv += state.numpy()[0]
            total_ld += state.numpy()[1]

            if done:  # update global and assign to local net
                Qval, _ = self.lnet.forward(new_state)
                Qval = Qval.detach().numpy()[0]
                all_rewards.append(np.sum(rewards))
                self.res_queue.put({'Reward': np.sum(rewards)})
                all_lengths.append(i)
                average_lengths.append(np.mean(all_lengths[-10:]))
                # print("episode: {}, reward: {}, total length: {}, average length: {} \n".format(self.name, np.sum(rewards), i, average_lengths[-1]))

                # compute Q values
                Qvals = np.zeros_like(values)
                for t in reversed(range(len(rewards))):
                    Qval = rewards[t] + GAMMA * Qval
                    Qvals[t] = Qval

                #update actor critic
                values = torch.FloatTensor(values)
                Qvals = torch.FloatTensor(Qvals)
                log_probs = torch.stack(log_probs)
                
                advantage = Qvals - values
                actor_loss = -(log_probs * advantage).mean()
                critic_loss = 0.5 * advantage.pow(2).mean()
                ac_loss = actor_loss + critic_loss - 0.01 * entropy_term

                self.opt.zero_grad()
                ac_loss.backward()
                for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                    gp.grad = lp.grad
                self.opt.step()

                self.lnet.load_state_dict(self.gnet.state_dict())
            
                log_probs = []
                values = []
                rewards = []

        sc = (total_pv + total_export)/total_pv
        ss = (total_pv + total_export)/total_ld
        print('sc and ss of ', self.name, sc, ss)
        self.res_queue.put({'SCSS':[sc, ss]})
    
        self.res_queue.put({'End':True})
        
def test(local_path, net):
    # customers = sorted(df.columns.levels[0])
    # data_test = df[1][['GG', 'GC']]
    # data_test['GC'].values[1]
    data_test, df_date = load_test_data(local_path)

    df_out = pd.DataFrame(columns=['PV', 'LD', 'PV.C', 'PV.D', 'BT', 'GD', 'COST', 'AC', 'RD'])
    MAX_EP = data_test.shape[0]

    env = Environment(data_test, 50.)
    
    with torch.no_grad():
        state = env.reset()
        for i in range(MAX_EP):
            _, logits = net.forward(state)
            action = torch.argmax(logits.unsqueeze(0), dim=1).numpy()[0]
            new_state, reward, done = env.step(action)
            # print('Action : ', a)

            st = np.concatenate((new_state, np.array([action-1, reward.squeeze(0)])))
            df_out.loc[i] = st    
            state = new_state

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
    local_path = os.getcwd()
    if local_path.split('/')[-1] == 'etc_project':
        local_path = os.path.join(local_path, 'proc_auggrid')
        
    data_train = load_train_data(local_path)
    
    gnet = a3c(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = torch.optim.Adam(gnet.parameters(), lr=1e-4)
    
    MPATH = os.path.join(local_path, '__pycache__/a3c_gnet.pt')
    if os.path.exists(MPATH):
        gnet.load_state_dict(torch.load(MPATH))

    res = []                    # record episode reward to plot
    scss = []
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, data_train[i]) for i in range(len(data_train))]
    [w.start() for w in workers]
    running_workers = len(workers)
    while running_workers > 0:
        r = res_queue.get()
        for k, v in r.items():
            if k == 'Reward':
                res.append(v)
            elif k == 'SCSS':
                print('SCSS', v)
                scss.append(v)
            elif k == 'End':
                print('End')
                running_workers -= 1
                
    [w.join() for w in workers]

    torch.save(gnet.state_dict(), MPATH)

    gnet.load_state_dict(torch.load(MPATH, weights_only=True))

    # test(local_path, gnet)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 4))
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

    print(scss)
    plt.plot(np.array(scss)[:, 0])
    plt.plot(np.array(scss)[:, 1])
    plt.show()
