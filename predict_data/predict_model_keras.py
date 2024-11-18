import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

SCR_DATA = 'Ppv_actual_synthetic_data'

def train(name):
	df = pd.read_csv(f"{SCR_DATA}.csv", encoding = "utf8")
	print(df)

	label_data = df.drop('1900-01-01 00:00:00', axis=1)
	print(label_data)
	# label_data = pd.concat([label_data, df.tail(1)])
	# print(label_data)

	x_train, x_test, y_train, y_test = train_test_split(df, label_data, test_size=0.3, shuffle=True)
	x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5,
													random_state = 1, shuffle = False)

	NF = len(df.columns)
	wfm = Sequential()
	wfm.add(Dense(160*NF, input_dim = NF, activation = 'tanh'))
	wfm.add(Dense(80*NF, activation = 'tanh'))
	wfm.add(Dense(40*NF, activation = 'relu'))
	wfm.add(Dense(40*NF, activation = 'relu'))
	wfm.add(Dense(20*NF, activation = 'relu'))
	wfm.add(Dense(20*NF, activation = 'relu'))
	wfm.add(Dense(NF-1, activation = 'relu'))
	wfm.add(Dense(NF-1, activation = 'relu'))

	wfm.summary()

	wfm.compile(loss = 'mean_squared_logarithmic_error', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

	early_stopping = EarlyStopping(monitor = 'loss', patience = 1000)

	wfm.fit(x_train, y_train, epochs = 10, batch_size = 10,
			validation_data = (x_val, y_val))
	wfm.save(name)

def predict(name):
	df = pd.read_csv(f"{SCR_DATA}.csv", encoding = "utf8")
	label_data = df.drop(df.index[0])
	label_data = pd.concat([label_data, df.tail(1)])
	NF = len(df.columns)
 
	model = load_model(name)
	start_seq = np.random.randint(0, NF-1)
	for i in range(10):
		pattern = df.loc[i, :].values.tolist()
		# pattern = np.zeros(NF)
		# pattern[1] = 15
		# pattern[2] = 25
		pre_in = np.reshape(pattern, (1, len(pattern), 1))
		result = model.predict(pre_in)
		print(f'Predict......: {result}')
		plt.plot(pattern, label='Origin')
		plt.plot(result[0], label='Predict')
		plt.show()
		
name = 'Predict_LSTM.h5' 
train(name)
predict(name)