import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

def getCities(dataset, input):
    cities = []
    for row in range(len(dataset.index)):
        city = dataset.at[row, "City"]
        if (city == "Philadelphia"):
            cities.append([1, 0, 0])
        elif (city == "Boston"):
            cities.append([0, 1, 0])
        elif (city == "Atlanta"):
            cities.append([0, 0, 1])
        else:
            cities.append([0, 0, 0])
    return np.c_[input, cities]


dataset = pd.read_csv("bigquery-geotab-intersection-congestion/train.csv")

x = dataset.filter(["Hour", "Month", "Weekend"], axis = 1).values
yDatasetLabels = ["TotalTimeStopped_p20", "TotalTimeStopped_p50", "TotalTimeStopped_p80", "DistanceToFirstStop_p20", "DistanceToFirstStop_p50", "DistanceToFirstStop_p80"]

yDataset = []
for label in yDatasetLabels:
    #placeholder1, x, placeholder2, y = train_test_split(inpX, dataset.filter([label], axis = 1).values[:,0], test_size=0.00001, random_state=0)
    y = dataset.filter([label], axis = 1).values[:,0]
    yDataset.append(y)

print("starting cities")
x = getCities(dataset, x)

print(x)

print("Fitting", yDataset)
regressors = []
hourPolys = []
monthPolys = []
for y in yDataset:
    print(len(y),len(x[:,0]))
    hourPoly = PolynomialFeatures(degree = 4)
    hourX = hourPoly.fit_transform(x, y=y)
    hourPolys.append(hourPoly)

    regressor = LinearRegression()
    regressor.fit(hourX, y)
    regressors.append(regressor)

testInp = pd.read_csv("bigquery-geotab-intersection-congestion/test.csv")
test = testInp.filter(["Hour", "Month", "Weekend"], axis = 1)
test = getCities(testInp, test)

print("Starting Labels")
resultLabels = [None] * (len(testInp.index) * 6)
position = 0
for j in range(6):
    for row in range(len(testInp.index)):
        if (row % 10000 == 0):
            print("\r%s" % (str((j + row * 1.0 / len(testInp.index)) / 6.0)), end="")
        resultLabels[position] = "".join([str(testInp.at[row, "RowId"]), "_", str(j)])
        position += 1
print("")

print("Prediction")
result = []
for j in range(6):
    print(j)
    testX = hourPolys[j].transform(test)
    prediction = regressors[j].predict(testX)
    result.append(prediction)

result = pd.DataFrame(data={"TargetId": resultLabels, "Target" : np.concatenate(result, axis=None)})
result.loc[result["Target"] < 0, ["Target"]] = 0
result.to_csv("./output.csv", index=False)
#print(dataset.loc[dataset['City'] == "test"])
