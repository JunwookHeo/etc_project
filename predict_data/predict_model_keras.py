import numpy as np
import tensorflow as tf

import pandas as pd

from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

SCR_DATA = 'Ppv_actual_synthetic_data'

df = pd.read_csv(f"{SCR_DATA}.csv", encoding = "utf8")
print(df)