import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Build training set from data
trainingDataFile = open("facebook_comment_volume_dataset/Dataset/Training/Features_Variant_1.csv")
datalines = trainingDataFile.readlines()

trainingXMatrix = []
trainingYMatrix = []

for line in datalines:
    row = [float(i) for i in line.strip().split(",")]
    trainingXMatrix.append(row[:-1])
    trainingYMatrix.append(row[-1])

# Data normalization
scaler = StandardScaler().fit(trainingXMatrix)
stxm = scaler.transform(trainingXMatrix)

neuralNet = MLPRegressor(hidden_layer_sizes=[100, 100], solver="adam", n_iter_no_change=100, max_iter=400, activation="logistic", verbose=True).fit(stxm, trainingYMatrix)

plt.plot([i for i in range(len(neuralNet.loss_curve_))], neuralNet.loss_curve_)
plt.show()