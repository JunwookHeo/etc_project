import pandas as pd
import numpy as np
import glob

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer

SRC_PATH = "./syntheic_data/archive/*.csv"

def preprocessing(field):
    src_files = glob.glob(SRC_PATH)

    dst_df = pd.DataFrame()
    for f in src_files:
        print(f) 
        df = pd.read_csv(f)        
        df['Time_actual'] = pd.to_datetime(df['Time_actual'], format='%Y-%m-%d %H:%M:%S')
        df['Time'] = df['Time_actual'].dt.strftime("%H:%M:%S")
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
        print(df.head())
        print(df.dtypes)
        # df.Time_actual = pd.to_datetime(df.Time_actual).dt.strftime('%H:%M')
        df = df.groupby(pd.Grouper(key='Time', freq='30min')).mean().dropna()          
        dst_df = dst_df._append(df[field].T)

    dst_df.reset_index(drop=True, inplace=True)
    dst_df.fillna(value=np.nan, inplace=True)
    dst_df.fillna(0, inplace=True)
    dst_df.to_csv(f'{field}.csv', index=False)
    return dst_df

def train_synthetic_model(field):
    df = pd.read_csv(f'{field}.csv')
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.to_dict()

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)

    synthesizer = CTGANSynthesizer(
        metadata, # required
        enforce_rounding=False,
        epochs=10000,
        verbose=True
    )

    synthesizer.get_parameters()
    metadata = synthesizer.get_metadata()
    synthesizer.fit(df)
    synthesizer.save(
        filepath=f'{field}.pkl'
    )

def gen_synthetic_data(field):
    synthesizer = CTGANSynthesizer.load(
        filepath=f'{field}.pkl'
    )

    synthetic_data = synthesizer.sample(num_rows=200)
    synthetic_data.to_csv(f'./{field}_synthetic_data.csv', encoding='utf8', index=False)

fields = ["Ppv_actual", "Pload_actual"]
for field in fields:
    df = preprocessing(field)
    train_synthetic_model(field)
    gen_synthetic_data(field)