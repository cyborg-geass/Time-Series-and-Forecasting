import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
warnings.filterwarnings("ignore")

train_data = pd.read_csv("Train_SU63ISt.csv", encoding="UTF-8")
test_data = pd.read_csv("Test_0qrQsBZ.csv", encoding="UTF-8")

print(train_data.columns, test_data.columns)
# print(train_data.head(), test_data.head())
train_orig = train_data.copy(True)
test_orig = test_data.copy(True)
# print(train_orig.shape, test_orig.shape)

train_data['Datetime'] = pd.to_datetime(train_data["Datetime"], format="%d-%m-%Y %H:%M")
test_data['Datetime'] = pd.to_datetime(test_data["Datetime"], format="%d-%m-%Y %H:%M")
train_orig['Datetime'] = pd.to_datetime(train_orig["Datetime"], format="%d-%m-%Y %H:%M")
test_orig['Datetime'] = pd.to_datetime(test_orig["Datetime"], format="%d-%m-%Y %H:%M")

for i in (train_data, test_data, train_orig, test_orig):
    i['Year'] = i['Datetime'].dt.year
    i['Month'] = i["Datetime"].dt.month
    i['Day'] = i["Datetime"].dt.day
    i['Hour'] = i["Datetime"].dt.hour

train_data['Day of Week'] = train_data["Datetime"].dt.dayofweek
temp = train_data['Datetime']
def applyer(row):
    if row.dayofweek==5 or row.dayofweek==6:
        return 1
    else:
        return 0
temp2 = train_data["Datetime"].apply(applyer)
train_data['Weekend'] = temp2
train_data.index = train_data["Datetime"]
df = train_data.drop("ID", axis=1)
ts = df['Count']
plt.figure(figsize=(16,8))
plt.plot(ts, label="passenger count")
plt.xlabel("Time(year-month")
plt.ylabel("Passenger Count")
plt.legend(loc = 'best')
train_data.groupby('Year')['Count'].mean().plot.bar()
# time.sleep(3)
temp3 = train_data.groupby(['Year', 'Month'])['Count'].mean()
temp3.plot(figsize=(15, 5), title="Passenger Count(MonthWise)", fontsize=14)
train_data.groupby('Day')['Count'].mean().plot.bar()
# print(df)
train_data = train_data.drop('ID', axis=1)
# train_data.timestamp = pd.to_datetime(train_data.Datetime, format="%d-%m-%Y %H:%M")
# train_data.index = train_data.timestamp
test_data.timestamp = pd.to_datetime(test_data["Datetime"], format="%d-%m-%Y %H-%M")
test_data.index = test_data.timestamp

test_data = test_data.resample('D').mean()
train_data =train_data.resample('D').mean()

train = train_data[:650]
# valid = train_data.ix['2014-06-25':'2014-09-25']

print(train)