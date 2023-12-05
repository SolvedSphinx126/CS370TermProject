import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from sklearn import linear_model
from sklearn.svm import LinearSVR
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
validationDataFile = open("facebook_comment_volume_dataset/Dataset/Testing/TestSet/Test_Case_7.csv")
# validationDataFile = open("facebook_comment_volume_dataset/Dataset/Testing/Features_TestSet.csv")

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
        keras.layers.Input(shape=(len(xTrainScaled[0]),), name="inputLayer"),
        keras.layers.Dense(128, activation="sigmoid", name="hiddenLayer1", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2()),
        keras.layers.Dense(64, activation="sigmoid", name="hiddenLayer2", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2()),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="sigmoid", name="hiddenLayer3", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2()),
        keras.layers.Dense(1, activation="linear", name="outputLayer", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2())
    ]
)
ANNModel.summary()

ANNModel.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.MeanAbsoluteError()])

epochs = 80 # IMPORTANT this is the number of epochs for the ANN model

history = ANNModel.fit(x=xTrainScaled, y=trainingYMatrix, epochs=epochs, batch_size=200, validation_data=(xTestScaled, testingYMatrix), verbose=1)

# Plotting Loss
plt.figure("ANN Convergence")
plt.subplot(2, 1, 1)
plt.plot([i for i in range(epochs)], history.history["loss"], color="red", label="Training Loss")
plt.plot([i for i in range(epochs)], history.history["val_loss"], color="green", label="Validation Loss")
plt.title("ANN: Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plotting Mean Abs Error
plt.subplot(2, 1, 2)
plt.plot([i for i in range(epochs)], history.history["mean_absolute_error"], color="red", label="Training MAE")
plt.plot([i for i in range(epochs)], history.history["val_mean_absolute_error"], color="green", label="Validation MAE")
plt.title("ANN: Training and validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplots_adjust(hspace=0.5)

mse_neural, mae_neural = ANNModel.evaluate(xTestScaled, testingYMatrix)

print("Mean squared error for Artificial Neural Net: ", mse_neural)
print("Mean absolute error for Artificial Neural Net: ", mae_neural)


############################################################################################################################
# Linear Regression Closed Form
lr_model = linear_model.LinearRegression()
lr_model.fit(xTrainScaled, trainingYMatrix)
y_pred_lr = lr_model.predict(xTestScaled)
mse_lr = mean_squared_error(testingYMatrix, y_pred_lr)
mae_lr = mean_absolute_error(testingYMatrix, y_pred_lr)

print("Mean squared error for Closed Form Linear Regression: ", mse_lr)
print("Mean absolute error for Closed Form Linear Regression: ", mae_lr)

############################################################################################################################
# Linear Regression SGD
lr_sgd_model = linear_model.SGDRegressor(loss="huber", penalty=None, verbose=False)

sgdTrainingLoss = []
sgdTestingLoss = []

epochs = 200
for i in range(epochs):
    lr_sgd_model.partial_fit(xTrainScaled, trainingYMatrix)

    sgdTrainingLoss.append(mean_squared_error(lr_sgd_model.predict(xTrainScaled), trainingYMatrix))
    sgdTestingLoss.append(mean_squared_error(lr_sgd_model.predict(xTestScaled), testingYMatrix))

# Plotting Loss (MSE)
plt.figure("LR Convergence")
plt.plot([i for i in range(epochs)], sgdTrainingLoss, color="red", label="Training Loss")
plt.plot([i for i in range(epochs)], sgdTestingLoss, color="green", label="Validation Loss")
plt.title("SGD_LR: Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

y_pred_sgd_lr = lr_sgd_model.predict(xTestScaled)
mse_sgd_lr = mean_squared_error(testingYMatrix, y_pred_sgd_lr)
mae_sgd_lr = mean_absolute_error(testingYMatrix, y_pred_sgd_lr)

print("Mean squared error for SGD Linear Regression: ", mse_sgd_lr)
print("Mean absolute error for SGD Linear Regression: ", mae_sgd_lr)

############################################################################################################################
# Support Vector Regression
svr_model = LinearSVR(verbose=True)
svr_model.fit(xTrainScaled, trainingYMatrix)
y_pred_svr = svr_model.predict(xTestScaled)
mse_svr = mean_squared_error(testingYMatrix, y_pred_svr)
mae_svr = mean_absolute_error(testingYMatrix, y_pred_svr)

print("Mean squared error for Support Vector Regression: ", mse_svr)
print("Mean absolute error for Support Vector Regression: ", mae_svr)