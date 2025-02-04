import pandas as pd
import numpy as np
import math
import torch

################################################################################
# Load training/test data from csv
################################################################################
def load_data(path, samples, TS=48):
    df = pd.read_csv(path, header=[0,1], index_col=0)
    df = df.set_index(pd.to_datetime(df.index))
    df.columns = df.columns.set_levels(df.columns.levels[0].astype('int64'), level=0)
    df = df/2000.
    df_date = df.index
    df.head()

    customers = sorted(df.columns.levels[0])
    data_train = []
    # samples = list(range(1, 2)) #[201, 202, 203]

    for s in samples:
        try:
            train = df[s][['GG', 'GC']]
            train['GC'].values[1]
            # print(train.shape)
            data_train.append(train)
        except:
            print(s, "An exception occurred.")
            data_train.append(pd.DataFrame())

    return data_train, df_date

################################################################################
# Calculate std deviation from daily usage 
################################################################################
def cal_stds(df, cmin=0.10, cmax=0.95):
    df_daily = df.groupby(df.index.date).sum()
    df_daily['Diff'] = df_daily['GG'] - df_daily['GC']
    d_68 = df_daily['Diff'].mean()+1.0*df_daily['Diff'].std()
    d_86 = df_daily['Diff'].mean()+1.5*df_daily['Diff'].std()
    d_95 = df_daily['Diff'].mean()+2.0*df_daily['Diff'].std()
    d_99 = df_daily['Diff'].mean()+3.0*df_daily['Diff'].std()
    
    w = 1/(cmax - cmin)

    return math.ceil(d_68*w), math.ceil(d_86*w), math.ceil(d_95*w), math.ceil(d_99*w)

################################################################################
# BATTERY model Environment for RL
################################################################################
class ENV_BATT:
    def __init__(self, data_env, capacity=10., horizon=48, maxsoc=0.95, minsoc=0.10):
        self.capacity = capacity
        self.data_env = data_env
        self.pos = 0
        self.horizon = horizon
        self.btmax = capacity*maxsoc
        self.btmin = capacity*minsoc
        self.state = None

        self.ss = np.zeros((4, int(self.horizon))) #pv, ld, import, export
        
    def reset(self):
        self.ss = np.zeros(self.ss.shape)

        self.pos = 0        
        self.state = np.array([0., 0., 0., 0., self.btmin, 0., 0])
        return torch.as_tensor(self.state, dtype=torch.float32).squeeze(0)

    def step(self, action):
        charging = 0.0
        discharging = 0.0
        batt_state = 0.0
        grid_state = 0.0
        reward = 0.
        wt = 0.5 # weight of sc and ss

        self.state[0] = self.data_env['GG'].values[self.pos]  # PV generation power
        self.state[1] = self.data_env['GC'].values[self.pos]  # Load consumption power
        
        if action == 1: # charge battery
            charging = max(0, self.state[0] - self.state[1])  # Charging power
            charging = min(self.btmax - self.state[4], charging)
            batt_state = charging # Update battery status
        elif action == 0: # discharge battery
            discharging = min(max(self.state[4]-self.btmin, 0), max(0, self.state[1] - self.state[0])) # discharing power
            batt_state = -discharging # Update battery status

        grid_state = self.state[1] - (self.state[0] - charging + discharging) # Grid power + : import, - : export

        C_A = 0.01
        C_B = 1.
        C_C = 0.
        cost = C_A*grid_state**2 + C_B*abs(grid_state) + C_C
        cost = cost if grid_state > 0 else -cost
                
        self.state[2] = charging
        self.state[3] = discharging
        self.state[4] += batt_state
        self.state[5] = grid_state
        self.state[6] = cost

        self.ss[0][self.pos%self.horizon] = self.state[0]
        self.ss[1][self.pos%self.horizon] = self.state[1]
        self.ss[2][self.pos%self.horizon] = grid_state if grid_state > 0 else 0 # import from Grid
        self.ss[3][self.pos%self.horizon] = abs(grid_state) if grid_state < 0 else 0 # export to Grid

        sc = (self.ss[0].sum() - self.ss[3].sum())/self.ss[0].sum() if self.ss[0].sum() else 0
        ss = (self.ss[1].sum() - self.ss[2].sum())/self.ss[1].sum() if self.ss[1].sum() else 0
        reward = wt*sc + (1-wt)*ss

        done = False
        self.pos += 1

        return torch.as_tensor(self.state, dtype=torch.float32), torch.tensor([reward]), done 

    def render(self):
        return 0