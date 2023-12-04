import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras as keras
from keras import layers

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Build training set from data
trainingDataFile = open("facebook_comment_volume_dataset/Dataset/Training/Features_Variant_1.csv")

trainingXMatrix = []
trainingYMatrix = []

for line in trainingDataFile.readlines():
    row = [float(i) for i in line.strip().split(",")]
    trainingXMatrix.append(row[:-1])
    trainingYMatrix.append(row[-1])

trainingXMatrix = np.array(trainingXMatrix)
trainingYMatrix = np.array(trainingYMatrix)

# Bulid test set from data
validationDataFile = open("facebook_comment_volume_dataset/Dataset/Testing/TestSet/Test_Case_2.csv")

testingXMatrix = []
testingYMatrix = []

for line in validationDataFile.readlines():
    row = [float(i) for i in line.strip().split(",")]
    testingXMatrix.append(row[:-1])
    testingYMatrix.append(row[-1])

testingXMatrix = np.array(testingXMatrix)
testingYMatrix = np.array(testingYMatrix)

# Data normalization
scaler = StandardScaler().fit(trainingXMatrix)
xTrainScaled = scaler.transform(trainingXMatrix)
xTestScaled = scaler.transform(testingXMatrix)

print("Size of input layer:" + str(len(xTrainScaled[0])))

############################################################################################################################
# Artificial Neural Network
ANNModel = keras.Sequential(
    [
        layers.Input(shape=(len(xTrainScaled[0]),), name="inputLayer"),
        layers.Dense(128, activation="sigmoid", name="hiddenLayer1"),
        layers.Dense(64, activation="sigmoid", name="hiddenLayer2"),
        # layers.Dense(4, activation="sigmoid", name="hiddenLayer3"),
        layers.Dense(1, activation="linear", name="outputLayer")
    ]
)
ANNModel.summary()

ANNModel.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.MeanAbsoluteError()])

epochs = 100

history = ANNModel.fit(x=xTrainScaled, y=trainingYMatrix, epochs=epochs, batch_size=200, validation_data=(xTestScaled, testingYMatrix))

# Plotting Loss
plt.plot([i for i in range(epochs)], history.history["loss"], color="red", label="Training Loss")
plt.plot([i for i in range(epochs)], history.history["val_loss"], color="green", label="Validation Loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotting Mean Abs Error
plt.plot([i for i in range(epochs)], history.history["mean_absolute_error"], color="red", label="Training MAE")
plt.plot([i for i in range(epochs)], history.history["val_mean_absolute_error"], color="green", label="Validation MAE")
plt.title("Training and validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

mse_neural, mae_neural = ANNModel.evaluate(xTestScaled, testingYMatrix)

print("Mean squared error for Artificial Neural Net: ", mse_neural)
print("Mean absolute error for Artificial Neural Net: ", mae_neural)


############################################################################################################################
# Linear Regression
lr_model = linear_model.LinearRegression()
lr_model.fit(xTrainScaled, trainingYMatrix)
y_pred_lr = lr_model.predict(xTestScaled)
mse_lr = mean_squared_error(testingYMatrix, y_pred_lr)
mae_lr = mean_absolute_error(testingYMatrix, y_pred_lr)

print("Mean squared error for Linear Regression: ", mse_lr)
print("Mean absolute error for Linear Regression: ", mae_lr)