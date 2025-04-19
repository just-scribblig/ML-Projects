import pandas as pd
dataset = pd.read_csv('smart_irrigation_dataset.csv')
X = dataset.iloc[:, :-2]
Y = dataset.iloc[:, -1]

# To see the features and targets
print("Features used: ", dataset.columns[:-2].str.strip().tolist()) #for feature columns, It gives Temperature and Humidity
print("Target used: ", dataset.columns[-1]) #target column = water_level

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
X_train, X_test, Y_train, Y_test = X_train.values, X_test.values, Y_train.values, Y_test.values

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
Y_pred = linear_model.predict(X_test)

print("Prdicted: ", Y_pred)
print("Expected: ", Y_test)

from sklearn import metrics
import numpy as np
print("Root Mean Absolute Error: ", np.sqrt(metrics.mean_absolute_error(Y_test, Y_pred)))
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print("R\u00B2 Score: ", metrics.r2_score(Y_test, Y_pred))
