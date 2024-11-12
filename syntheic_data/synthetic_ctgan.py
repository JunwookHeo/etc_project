import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

df = pd.read_csv("test.csv")

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(df)

metadata.to_dict()

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df)
synthetic_data = synthesizer.sample(num_rows=100)

from sdv.single_table import CTGANSynthesizer

# 모델 정의
synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=100,
    verbose=True
)


synthesizer.get_parameters()
metadata = synthesizer.get_metadata()

# 생성 모델 학습
synthesizer.fit(df)

# 합성 데이터 생성
synthetic_data = synthesizer.sample(num_rows=100000)

# 모델 저장
synthesizer.save(
    filepath='my_synthesizer.pkl'
)