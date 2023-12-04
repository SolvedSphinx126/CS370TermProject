import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def getMeanAbsError(predicted, actual):
    total = 0
    for index, val in enumerate(predicted):
        total += abs(val - actual[index])

    return total / len(predicted)


# Build training set from data
trainingDataFile = open("facebook_comment_volume_dataset/Dataset/Training/Features_Variant_1.csv")

trainingXMatrix = []
trainingYMatrix = []

for line in trainingDataFile.readlines():
    row = [float(i) for i in line.strip().split(",")]
    trainingXMatrix.append(row[:-1])
    trainingYMatrix.append(row[-1])

# Bulid test set from data
validationDataFile = open("facebook_comment_volume_dataset/Dataset/Testing/TestSet/Test_Case_2.csv")

testingXMatrix = []
testingYMatrix = []

for line in validationDataFile.readlines():
    row = [float(i) for i in line.strip().split(",")]
    testingXMatrix.append(row[:-1])
    testingYMatrix.append(row[-1])

# Data normalization
scaler = StandardScaler().fit(trainingXMatrix)
stxm = scaler.transform(trainingXMatrix)
svxm = scaler.transform(testingXMatrix)

neuralNet = MLPRegressor(hidden_layer_sizes=[100, 100], solver="adam", n_iter_no_change=100, max_iter=200, batch_size=200, activation="logistic", verbose=True).partial_fit(stxm, trainingYMatrix)
nIter = 50

# Manually iterate the ANN and record the training and testing error
trainingCosts = []
validingCosts = []
for i in range(nIter):
    neuralNet.partial_fit(stxm, trainingYMatrix)
    trainingCosts.append(getMeanAbsError(neuralNet.predict(stxm), trainingYMatrix))
    validingCosts.append(getMeanAbsError(neuralNet.predict(svxm), testingYMatrix))
    print(i)

plt.plot([i for i in range(nIter)], trainingCosts, color="red")
plt.plot([i for i in range(nIter)], validingCosts, color="green")
plt.show()


# Predict values
predictedYMatrix = neuralNet.predict(svxm)

print("Testing M.A.E: " + str(getMeanAbsError(predictedYMatrix, testingYMatrix)))
print("Training M.A.E: " + str(getMeanAbsError(neuralNet.predict(stxm), trainingYMatrix)))
# for index, val in enumerate(predictedYMatrix):
    # print("Actual: " + str(testingYMatrix[index]) + " = " + str(val) + " :Predicted")