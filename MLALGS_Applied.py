import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

import time
import math

from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

PRINT_DETAILS = False

#####################
# Build training sets
numVariants = 5
trainingXList = []
trainingYList = []
for i in range(numVariants):
    trainingDataFile = open(f"facebook_comment_volume_dataset/Dataset/Training/Features_Variant_{i + 1}.csv")

    trainingXMatrix = []
    trainingYMatrix = []

    for line in trainingDataFile.readlines():
        row = [float(i) for i in line.strip().split(",")]
        trainingXMatrix.append(row[:-1])
        trainingYMatrix.append(row[-1])

    trainingXMatrix = np.array(trainingXMatrix)
    trainingYMatrix = np.array(trainingYMatrix)

    trainingXList.append(trainingXMatrix)
    trainingYList.append(trainingYMatrix)

####################
# Build testing sets
numTestCases = 10
testingXList = []
testingYList = []
for i in range(numTestCases):
    validationDataFile = open(f"facebook_comment_volume_dataset/Dataset/Testing/TestSet/Test_Case_{i + 1}.csv")

    testingXMatrix = []
    testingYMatrix = []

    for line in validationDataFile.readlines():
        row = [float(i) for i in line.strip().split(",")]
        testingXMatrix.append(row[:-1])
        testingYMatrix.append(row[-1])

    testingXMatrix = np.array(testingXMatrix)
    testingYMatrix = np.array(testingYMatrix)

    testingXList.append(testingXMatrix)
    testingYList.append(testingYMatrix)

def GetHits(predictions, actual, numPostsToConsider=10, debugPrint=0):
    realTopPosts = sorted(enumerate(actual), key=lambda x: x[1], reverse=True)[:numPostsToConsider]
    predictedTopPosts = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:numPostsToConsider]

    realIndexes = [index for index, value in realTopPosts]
    predictedIndexes = [index for index, value in predictedTopPosts]

    hits = len(set(realIndexes).intersection(predictedIndexes))

    if (debugPrint):
        print(f"Real top {numPostsToConsider} posts: {realTopPosts}")
        print(f"Predicted top {numPostsToConsider} posts: {predictedTopPosts}")
        print(f"Hits: {hits}")

    return hits

def GetAUC(predictions, actual, numPostsToConsider=10, debugPrint=0):
    hits = GetHits(predictions, actual, numPostsToConsider, debugPrint)
    return hits / (hits + (numPostsToConsider - hits))

def GetMAE(predictions, actual):
    errors = abs(actual - predictions)

    MeanAE = np.mean(errors)
    return MeanAE

# Returns the metrics of the model trained on the specified data for each of the 10 test cases
def GetBatchMetrics(model, trainingXList, trainingYList, testCases, testCasesYs, epochs = None, printCrap = 0, batchSize=200):

    metrics = {}

    for index, trainingX in enumerate(trainingXList):
        scaler = StandardScaler().fit(trainingX)
        sTrainingX = scaler.transform(trainingX)

        trainingY = trainingYList[index]

        startTime = time.time()

        if (type(model) == keras.Sequential):
            model.fit(x=sTrainingX, y=trainingY, epochs=epochs, verbose=printCrap, batch_size=batchSize)
        elif (type(model) == linear_model.SGDRegressor):
            model.fit(sTrainingX, trainingY) # Max iter is set as a property of this model
        elif (type(model) == LinearSVR):
            model.fit(sTrainingX, trainingY) # Max iter is set as a property of this model
        else:
            return

        predictions = []

        hitsAt10 = []
        AUCAt10 = []
        MeanAE = []

        for j, testCaseX in enumerate(testCases):
            sTestCaseX = scaler.transform(testCaseX)
            predictions.append(model.predict(sTestCaseX))
            hitsAt10.append(GetHits(predictions[j], testCasesYs[j]))
            AUCAt10.append(GetAUC(predictions[j], testCasesYs[j]))
            MeanAE.append(GetMAE(predictions[j], testCasesYs[j]))

        endTime = time.time()

        caseMetrics = {}
        caseMetrics["HitsAt10"] = hitsAt10
        caseMetrics["AUCAt10"] = AUCAt10
        caseMetrics["MeanAE"] = MeanAE

        print(f"Completed variant {index + 1} of model {'ANN' if type(model) == keras.Sequential else 'LR_SGD' if type(model) == linear_model.SGDRegressor else 'SVR' if type(model) == LinearSVR else 'Error?'} in {endTime - startTime} seconds")
        metrics[f"Variant{index + 1}_TestMetrics"] = caseMetrics
        metrics[f"Variant{index + 1}_TotalTime"] = endTime - startTime
    
    return metrics

def GetAverageMetric(metricObj, caseMetric, variant):
    return np.average(metricObj[f"Variant{variant}_TestMetrics"][caseMetric])

metricsDict = {}

############################################################################################################################
# Artificial Neural Network
ANNModel = keras.Sequential(
    [
        keras.layers.Input(shape=(len(trainingXList[0][0]),), name="inputLayer"),
        # keras.layers.Dense(128, activation="sigmoid", name="hiddenLayer1", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2()),
        keras.layers.Dense(64, activation="sigmoid", name="hiddenLayer1", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2()),
        # keras.layers.Dense(64, activation="sigmoid", name="hiddenLayer2", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2()),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="sigmoid", name="hiddenLayer3", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2()),
        keras.layers.Dense(1, activation="relu", name="outputLayer", kernel_regularizer=keras.regularizers.L1(), activity_regularizer=keras.regularizers.L2())
    ]
)
ANNModel.summary()

ANNModel.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.MeanAbsoluteError()])

epochs = 40 # IMPORTANT this is the number of epochs for the ANN model

metrics = GetBatchMetrics(ANNModel, trainingXList, trainingYList, testingXList, testingYList, epochs=epochs, printCrap=PRINT_DETAILS, batchSize=200)
print(metrics)
metricsDict["ANN"] = metrics

############################################################################################################################
# Linear Regression SGD
lr_sgd_model = linear_model.SGDRegressor(loss="huber", penalty=None, verbose=PRINT_DETAILS)

metrics = GetBatchMetrics(lr_sgd_model, trainingXList, trainingYList, testingXList, testingYList)
print(metrics)
metricsDict["LR_SGD"] = metrics

############################################################################################################################
# Support Vector Regression
svr_model = LinearSVR(verbose=PRINT_DETAILS)

metrics = GetBatchMetrics(svr_model, trainingXList, trainingYList, testingXList, testingYList)
print(metrics)
metricsDict["SVR"] = metrics

############################################################################################################################
# Plotting Metrics
categories = ["ANN", "LR_SGD", "SVR"]
avgHitsAt10AllV = []
for i in range(numVariants):
    avgHitsAt10AllV.append([GetAverageMetric(metricsDict[categories[0]], "HitsAt10", i + 1),
                            GetAverageMetric(metricsDict[categories[1]], "HitsAt10", i + 1),
                            GetAverageMetric(metricsDict[categories[2]], "HitsAt10", i + 1)])

avgAUCAt10AllV = []
for i in range(numVariants):
    avgAUCAt10AllV.append([GetAverageMetric(metricsDict[categories[0]], "AUCAt10", i + 1),
                           GetAverageMetric(metricsDict[categories[1]], "AUCAt10", i + 1),
                           GetAverageMetric(metricsDict[categories[2]], "AUCAt10", i + 1)])

MAEAllV = []
for i in range(numVariants):
    MAEAllV.append([GetAverageMetric(metricsDict[categories[0]], "MeanAE", i + 1),
                    GetAverageMetric(metricsDict[categories[1]], "MeanAE", i + 1),
                    GetAverageMetric(metricsDict[categories[2]], "MeanAE", i + 1)])

evalTime = []
for i in range(numVariants):
    evalTime.append([metricsDict[categories[0]][f"Variant{i + 1}_TotalTime"],
                     metricsDict[categories[1]][f"Variant{i + 1}_TotalTime"],
                     metricsDict[categories[2]][f"Variant{i + 1}_TotalTime"],])

barWidth = 0.1
barPos = []
barPos.append(np.arange(len(categories)))
for i in range(numVariants - 1):
    barPos.append([x + barWidth for x in barPos[i]])

# Avg hits at 10
plt.figure("AvgHitsAt10")
for index in range(numVariants):
    plt.bar(barPos[index], avgHitsAt10AllV[index], barWidth, label=f"Variant{index + 1}")

plt.xlabel("ML Alg")
plt.ylabel("Hits")
plt.title("Average Hits At 10")
plt.yticks(np.arange(10))
plt.xticks([pos + (barWidth * math.floor(numVariants / 2)) for pos in barPos[0]], categories)
plt.legend()

# Avg AUC at 10
plt.figure("AvgAUCAt10")
for index in range(numVariants):
    plt.bar(barPos[index], avgAUCAt10AllV[index], barWidth, label=f"Variant{index + 1}")

plt.xlabel("ML Alg")
plt.ylabel("AUC")
plt.title("Average AUC At 10")
plt.yticks(np.arange(1, step=0.1))
plt.xticks([pos + (barWidth * math.floor(numVariants / 2)) for pos in barPos[0]], categories)
plt.legend()

# MAE
plt.figure("MAE")
for index in range(numVariants):
    plt.bar(barPos[index], MAEAllV[index], barWidth, label=f"Variant{index + 1}")

plt.xlabel("ML Alg")
plt.ylabel("MAE")
plt.title("MAE")
plt.xticks([pos + (barWidth * math.floor(numVariants / 2)) for pos in barPos[0]], categories)
plt.legend()

# Eval Time
plt.figure("EvalTime")
for index in range(numVariants):
    plt.bar(barPos[index], evalTime[index], barWidth, label=f"Variant{index + 1}")

plt.xlabel("ML Alg")
plt.ylabel("Seconds")
plt.title("Evaluation Time")
plt.xticks([pos + (barWidth * math.floor(numVariants / 2)) for pos in barPos[0]], categories)
plt.legend()

plt.show()