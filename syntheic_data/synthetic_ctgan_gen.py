# import pandas as pd

# from sdv.metadata import SingleTableMetadata
# from sdv.single_table import GaussianCopulaSynthesizer

from sdv.single_table import CTGANSynthesizer

synthesizer = CTGANSynthesizer.load(
    filepath='my_synthesizer.pkl'
)

synthetic_data = synthesizer.sample(num_rows=1000)

# 데이터 csv로 저장
synthetic_data.to_csv('./synthetic_data_train.csv', encoding='utf8', index=False)