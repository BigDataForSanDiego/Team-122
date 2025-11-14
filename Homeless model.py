import pandas as pd
import numpy as np
from math import floor
#from tensorflow.keras.layers import Dense, BatchNormalization, Reshape
#from tensorflow.keras import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from tensorflow import GradientTape as gtape
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(20))
model.add(BatchNormalization())
model.add(Dense(25,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(20, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(20, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(20, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(10, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))


data=pd.read_csv("local master.xlsx - master_with_counts.csv")

n_lag=8
for i in range(3,n_lag+1):
    data["count_lag_"+str(i)]=data["homeless_count"].shift(i)
"""
data["count_lag_1"]=data["homeless_count"].shift(1)
data["count_lag_2"]=data["homeless_count"].shift(2)
data["count_lag_3"]=data["homeless_count"].shift(3)
data["count_lag_4"]=data["homeless_count"].shift(4)
data["count_lag_5"]=data["homeless_count"].shift(5)
data["count_lag_6"]=data["homeless_count"].shift(6)
data["count_lag_7"]=data["homeless_count"].shift(7)
data["count_lag_8"]=data["homeless_count"].shift(8)
"""
data["median_rent_city_lag1"] = data["ZORI"].shift(1)
data["avg_temp_lag1"] = data["average_temp"].shift(1)
data["unemployment_rate_lag1"] = data["unemployment_rate"].shift(1)
data["evictions_lag1"] = data["evictions"].shift(1)
data["precipitation_lag1"] = data["precipitation"].shift(1)
data["cpi_lag1"] = data["cpi"].shift(1)
data["cpi_lag2"] = data["cpi"].shift(2)
data["cpi_lag3"] = data["cpi"].shift(3)
#data=data.drop(["average_temp","ZORI","unemployment_rate","evictions"],axis=1)
"""
temp = []
for i in range(len(data["homeless_count"])):
    if i==0:
        temp.append(0)
    else:
        temp.append((data["homeless_count"][i]-data["homeless_count"][i-1]))
data["homeless_count"] = temp
"""
data=data.drop("shelter_beds",axis=1)
data = data.drop("industrial_production",axis=1)

for col in data.keys():
        if col in data.columns:
            data[col] = data[col].infer_objects(copy=False)
            data[col] = data[col].interpolate()
for i in range(12):
    data=data.drop(i)
l = len(data["homeless_count"])

for i in range(l-n_lag,l):
    data=data.drop(i)
data["month"] = [int(i[-1]) for i in data["year_month"]]
data["year"] = [int(i[:4]) for i in data ["year_month"]]
data=data.drop("year_month",axis=1)
features = data.drop("homeless_count",axis=1).values
labels =data["homeless_count"].values
"""
train_test_split = 0.75
len_train = floor(train_test_split*len(features))
len_test = len(features)-len_train
left  = len_train//2
right = left+len_test
x_train = np.concatenate((np.array(features[:left]),np.array(features[right:])))
y_train = np.concatenate((np.array(labels[:left]),np.array(labels[right:])))
x_test = np.array(features[left:right])
y_test = np.array(labels[left:right])
"""
x_train, x_test, y_train, y_test=train_test_split(np.array(features),np.array(labels),test_size=0.20)

"""
x_train= np.array(features[:floor(len(features)*train_test_split)])
y_train = np.array(labels[:floor(len(labels)*train_test_split//1)])
y_test = np.array(labels[floor(len(labels)*train_test_split//1):])
x_test = np.array(features[floor(len(features)*train_test_split//1):])
"""
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
print(data.keys())

"""
model = GradientBoostingRegressor(n_estimators=100,min_samples_leaf=6,learning_rate=0.05,max_depth=7,subsample=0.7)
#scores = cross_val_score(model, np.array(features),np.array(labels), cv=5) # cv=5 for 5-fold

print(f"Individual fold scores: {scores}")
print(f"Mean accuracy: {np.mean(scores)}")
model.fit(x_train,y_train)
print(model.predict(x_test))
print(y_test)
"""


schedule = ExponentialDecay(0.05,decay_steps=2000,decay_rate=0.96)
opt = Adam(learning_rate=schedule)
model.compile(optimizer=opt, loss="MSE", metrics=["R2Score"])
model.fit(x_train,y_train,epochs=500,batch_size=5)
print(model.predict(x_test))
print(y_test)

"""
for i,j in zip(x_test,y_test):
    print(model.predict(x_test), y_test)
    
"""

#epochs=200,with hisstorical, batch_size=5, no added lags for cpi-best
#taking off homeless counts from features and adding cpi lags greatly increased variance and matched the 2023 peaks, replicated success by adding back homeless counts

    
#bestscore: 0.3-done by moving test to middle, epochs=200, batch_size=5, no lags for counts before 3, 0.2 init learning rate, 3 cpi lags

#bestscore-rate=0.05, MSE, decay=0.96, drop indpro, shelter_beds, cpi lags, count lags after 3, caused by RANDOMIZING TRAIN-TEST

#testing and scoring
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
#all_x = np.concatenate((x_train,x_test))
#all_y = np.concatenate((y_train,y_test))

predict = model.predict(x_test)
actual = y_test
print(predict)
print(actual)
print(r2_score(predict,actual))
plt.plot([i for i in range(len(predict))],predict,label="predict")
plt.plot([i for i in range(len(actual))],actual,label="actual")
plt.legend()
plt.show()

