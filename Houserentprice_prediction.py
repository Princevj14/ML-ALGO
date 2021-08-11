# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:15:56 2021

@author: P_VJ
"""

import pandas as pd
df = pd.read_csv("Downloads/house_rental_data.csv.txt")

X = df.iloc[:, 1:7].values
y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size= 0.30, random_state = 0 )


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
#print(y_pred)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = []

for k in range(2,15):
    model = KNeighborsRegressor(n_neighbors = k)
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)

    error = sqrt(mean_squared_error(y_test,y_predict))
    rmse.append(error)
    print(k,error)

graph = pd.DataFrame(rmse)
graph.plot()    
