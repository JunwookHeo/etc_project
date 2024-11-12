import pandas as pd 
import seaborn as sns
import matplotlib as plt
import numpy as np 
from synthpop import Synthpop

df = pd.read_csv("2021-11-14_data.csv")
df = df.drop('Time_actual', axis=1)
# df.astype(dtypes)
# df['Time_actual'] = pd.to_datetime(df['Time_actual'], format='%Y-%m-%d %H:%M:%S')
# Fonction de conversion
def convert_dict_types(input_dict):
    # Mapping des types
    type_mapping = {
        'int64': 'int',
        'int32': 'int',
        'float32': 'float',
        'float64': 'float',
    }

    # Création du nouveau dictionnaire
    new_dict = {k: type_mapping[str(v)] for k, v in input_dict.items()}
    return new_dict

dtypes = convert_dict_types(df.dtypes)
print(dtypes)

spop = Synthpop() 
spop.fit(df, dtypes) 


df_syn = spop.generate(10) # 1000개 재현 데이터 생성
df_syn.head()