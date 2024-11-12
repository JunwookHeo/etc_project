import pandas as pd
import numpy as np
import glob

SRC_PATH = "./archive/*.csv"

def preprocessing(field):
    src_files = glob.glob(SRC_PATH)

    dst_df = pd.DataFrame()
    for f in src_files:
        print(f) 
        df = pd.read_csv(f)
        dst_df = dst_df._append(df['Ppv_actual'].T)

    dst_df.reset_index(drop=True, inplace=True)
    dst_df.fillna(value=np.nan, inplace=True)
    dst_df.fillna(0, inplace=True)
    return dst_df

df = preprocessing('Ppv_actual')
df.to_csv('test.csv', index=False)
