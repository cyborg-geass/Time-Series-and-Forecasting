"""
for normal machine learning models: try to use sklearn.pipeline.make_pipeline 
"""

import torch
import torchvision
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import RidgeCV, LinearRegression, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor

## Model:1
model_1 = DecisionTreeRegressor(random_state=42)
model_1.fit(X, y) # apply
model_1.predict(X_test) #apply

## Model:2
model_2 = RandomForestRegressor(n_estimators=150, random_state=42)
model_2.fit(X,y)
model_2.predict(X_test)

## Model:3
model_3 = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, random_state=42, verbose=1, loss="mse")
model_3.fit(X, y) ## Pease change the learning rate from 0.5 to 0.05 and try
model_3.predict(X_test)

## Model :4
model_4 = LinearSVR(verbose=1, random_state= 42, loss= "epsilon_insensitive", max_iter=2000)
model_4.fit(X,y)
model_4.predict(X_test)

## MOdel: 5
model_5 = LinearRegression(normalize=False)
model_5.fit(X, y)
model_5.predict(X_test)

## Model: 6
model_6 = TweedieRegressor(power=1.0, alpha=1, max_iter=200) ## You can change the iterations and power value to 0,1,2,3 or (1,2)
model_6.fit(X, y)
model_6.predict(X_test)

## Model: 7
estimators = [
    ('lr', RidgeCV()),
    ('SVR', LinearSVR(random_state=42))
]
model_7 = StackingRegressor(estimators, final_estimator=RandomForestRegressor(100, random_state=42))
model_7.fit(X, y)
model_7.predict(X_test)

## MOdel: 8
model_8 = tf.keras.Sequential(
    tf.keras.Input(shape=()), ## Give the input size
    tf.keras.layers.Dense(units=64),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=1),
    # tf.keras.layers.Activation("relu")  ## maybe tanh
)
model_8.fit(X, y)
model_8.predict(X_test)

## Model: 9
model_9 = tf.keras.Sequential(
    tf.keras.layers.Input(shape=()), # Give the shape here
    tf.keras.layers.RNN(return_sequences=True), 
    tf.keras.layers.RNN(),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dense(units=1)
)

## Model: 10
model_10 = tf.keras.Sequential(
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu")),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation="tanh")),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu")),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu")),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
)
model_11 = tf.keras.Sequential(
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu")),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRUCell(128, activation="tanh", return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRUCell(64, activation="tanh")),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu")),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu")),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
)
model_10.compile(optimizer="Adam", loss = "mse", metrics= "Crossentropy") ## use for different models
