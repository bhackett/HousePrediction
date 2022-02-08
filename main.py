# https://iaviral.medium.com/house-prediction-model-linear-regression-249e6db5da38

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

url = r"https://raw.githubusercontent.com/aviralb13/codes/main/datas/house_prediction(lr).csv"

house = pd.read_csv(url)
# print(house.head())
features = ["condition", "grade", "yr_built", "floors", "sqft_living"]
x = house[features]
y = house['price']
# print(f"x: {x}  y: {y}")
print(y)

train_x, val_x, train_y, val_y = train_test_split(x,y)
linear_model = LinearRegression()
linear_model.fit(train_x,train_y)


val_predictions = linear_model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))

r = linear_model.predict([[3, 7, 1951, 2.0, 2570]])
print(r)

