import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("/Users/lukas/projects/Kaggle/BigQuery/bigquery-geotab-intersection-congestion/train.csv")

dataset.drop(["TotalTimeStopped_p20","TotalTimeStopped_p40","TotalTimeStopped_p50","TotalTimeStopped_p60","TotalTimeStopped_p80","TimeFromFirstStop_p20","TimeFromFirstStop_p40","TimeFromFirstStop_p50","TimeFromFirstStop_p60","TimeFromFirstStop_p80","DistanceToFirstStop_p20","DistanceToFirstStop_p40","DistanceToFirstStop_p50","DistanceToFirstStop_p60","DistanceToFirstStop_p80"], axis = 1)

print(dataset)
#print(dataset.loc[dataset['City'] == "test"])
