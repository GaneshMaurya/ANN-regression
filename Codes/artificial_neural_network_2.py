# Artificial Neural Network

# Importing the Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Importing the Dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the Input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Add a second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1))


# Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the Train set
ann.fit(X_train, y_train, epochs = 100)


# Making the predictions

# Predicting the Test set results
y_pred = ann.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

