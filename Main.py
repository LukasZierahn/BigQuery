import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("/Users/lukas/projects/Kaggle/BigQuery/bigquery-geotab-intersection-congestion/train.csv")

#X = dataset.filter(["TotalTimeStopped_p20","TotalTimeStopped_p40","TotalTimeStopped_p50","TotalTimeStopped_p60","TotalTimeStopped_p80","TimeFromFirstStop_p20","TimeFromFirstStop_p40","TimeFromFirstStop_p50","TimeFromFirstStop_p60","TimeFromFirstStop_p80","DistanceToFirstStop_p20","DistanceToFirstStop_p40","DistanceToFirstStop_p50","DistanceToFirstStop_p60","DistanceToFirstStop_p80"], axis = 1)
inpX = dataset.filter(["Hour", "Month", "Weekend"], axis = 1).values
x = inpX
yDatasetLabels = ["TotalTimeStopped_p20", "TotalTimeStopped_p50", "TotalTimeStopped_p80", "DistanceToFirstStop_p20", "DistanceToFirstStop_p50", "DistanceToFirstStop_p80"]

yDataset = []
for label in yDatasetLabels:
    #placeholder1, x, placeholder2, y = train_test_split(inpX, dataset.filter([label], axis = 1).values[:,0], test_size=0.00001, random_state=0)
    y = dataset.filter([label], axis = 1).values[:,0]
    yDataset.append(y)

regressors = []
hourPolys = []
monthPolys = []
for y in yDataset:
    print(len(y),len(x[:,0]))
    hourPoly = PolynomialFeatures(degree = 4)
    hourX = hourPoly.fit_transform(x, y=y)
    hourPolys.append(hourPoly)

    print("its this")
    print(hourX)
    print(y)

    regressor = LinearRegression()
    regressor.fit(hourX, y)
    regressors.append(regressor)

testInp = pd.read_csv("/Users/lukas/projects/Kaggle/BigQuery/bigquery-geotab-intersection-congestion/test.csv")
test = testInp.filter(["Hour", "Month", "Weekend"], axis = 1)


print("Starting Labels")
resultLabels = [None] * (len(test.index) * 6)
position = 0
for j in range(6):
    for row in range(len(test.index)):
        if (row % 10000 == 0):
            print((j + row * 1.0 / len(test.index)) / 6.0)
        resultLabels[position] = "".join([str(testInp.at[row, "RowId"]), "_", str(j)])
        position += 1

print("Finished Labels")
result = []
for j in range(6):
    print(j)
    testX = hourPolys[j].transform(test.values)
    prediction = regressors[j].predict(testX)
    result.append(prediction)

result = pd.DataFrame(data={"TargetId": resultLabels, "Target" : np.concatenate(result, axis=None)})
result.loc[result["Target"] < 0, ["Target"]] = 0
result.to_csv("./output.csv", index=False)
#print(dataset.loc[dataset['City'] == "test"])
