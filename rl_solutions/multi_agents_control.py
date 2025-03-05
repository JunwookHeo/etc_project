from multiprocessing import Process, Queue

import pandas as pd
import numpy as np
import utils
import os, time

class AgentBess:
    def __init__(self, rxq, txq, hh, df, c, cmin=0.10, cmax=0.95, cinit=0.10):        

        self.rxq = rxq
        self.txq = txq
        self.hh = hh
        self.df = df
        self.c = c
        self.cmin = c*cmin
        self.cmax = c*cmax
        self.soc = c*cinit

    def action(self, dt, gsoc):
        grid = 0
        charge = 0
        charge_gd = 0
        [p, l] = self.df.loc[[dt]].values[0]
        if p > l: # charging
            charge = min(self.cmax - self.soc, p - l)
            grid = -(p - l - charge)
        else: # discharging
            charge = -min(self.soc - self.cmin, l - p)
            grid = l - p + charge
        self.soc += charge

        # lsoc = self.soc/self.c
        # if lsoc < gsoc: #charge more power to soc from grid
        #     charge_gd = min(self.cmax - self.soc, (gsoc - lsoc)*self.c)
        #     grid += charge_gd
        # else: # export power from soc to grid
        #     charge_gd = -min(self.soc - self.cmin, (lsoc - gsoc)*self.c)
        #     grid += charge_gd

        # self.soc += charge_gd
        # charge += charge_gd
        
        return charge, grid

    def run(self):
        imp = 0
        exp = 0
        while True:
            data = self.rxq.get()
            if data['CMD'] == 'STOP': break
            elif data['CMD'] == 'DATE':
                charge, grid = self.action(data['VALUE'], data['GSOC'])
                if grid < 0: exp += grid
                else: imp += grid
                
                self.txq.put({'HH':self.hh, 'GD':grid, 'SOC':self.soc/self.c, 'CHG':charge})

        sc = 1 + exp/self.df['GG'].sum()
        ss = 1 - imp/self.df['GC'].sum()

        self.txq.put({'HH':self.hh, 'SC':sc, 'SS':ss})

def work(hh, agent):
    agent.run()
    return

if __name__ == "__main__":
    TS = 48

    local_path = os.getcwd()
    if local_path.split('/')[-1] == 'etc_project':
        local_path = os.path.join(local_path, 'rl_solutions')
        
    samples = [25, 93, 48, 75]
    # samples = [93, 48]

    df_sel, df_date = utils.load_data(os.path.join(local_path, 'AusGrid_preprocess.csv'), samples, TS)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    mainQ = Queue()
    agents = []
    agentQs = []

    df_con = pd.DataFrame(index=df_date)
    for df in df_sel:
        df_con = df.add(df_con, fill_value=0)
        

    for hh, df in zip(samples, df_sel):
        c = 10
        agentQ = Queue()
        agent = AgentBess(agentQ, mainQ, hh, df, c)
        thd = Process(target=work, args=(hh, agent))
        thd.start()
        agents.append(thd)
        agentQs.append(agentQ)

    gd_list = []
    g_soc = 0
    for d in df_date:
        for q in agentQs:
            q.put({'CMD':'DATE', 'VALUE':d, 'GSOC':g_soc})

        gd = 0
        g_soc = 0
        for agent in agents:
            data = mainQ.get()
            # print(f'{data["HH"]} {data["GD"]} {data["SOC"]}')
            gd += data['GD']
            g_soc += data['SOC']
        g_soc /= len(agents)
        gd_list.append(gd)
        
    for q in agentQs:
        q.put({'CMD':'STOP'})
    for agent in agents:
        data = mainQ.get()
        print(data)

    df_con['GD'] = np.array(gd_list).tolist()
    exp = df_con['GD'].where(df_con['GD'] < 0).sum()
    imp = df_con['GD'].where(df_con['GD'] > 0).sum()
    
    sc = 1 + exp/df_con['GG'].sum()
    ss = 1 - imp/df_con['GC'].sum()
        
    print(f"Finished : {sc} {ss}")

  
