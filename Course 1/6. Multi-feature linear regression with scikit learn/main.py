# ranges:
# size 852 - 2104
# bedrooms 2 - 5
# floors 1 - 2
# age of home 35 - 45
# price 178 - 460

# use standard feature scaling to make all features 0 <= x <= 1

# IMPORTANT: must normalise input data, and de normalise output prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler  # used for z score normalization

features_dataframe = pd.read_csv("House_features.csv")
feature_names = [name for name in features_dataframe.columns]

dataframe_lists = features_dataframe.values.tolist()  # turns each row into list

# ith row of x values from the csv data would be dataframe_lists[i][:-1]

y_data_list = [float(list[-1]) for list in dataframe_lists[0:]]
x_data_lists = [list[:-1] for list in dataframe_lists[0:]]

x_data_arrays = np.array(x_data_lists)
y_data_array = np.array(y_data_list)

print(y_data_array, x_data_arrays)

# use z score normalization on x data:

z_scalar = StandardScaler()
x_normalised = z_scalar.fit_transform(x_data_lists)

# create the regression object
regression = SGDRegressor()
regression.fit(x_normalised, y_data_array)
print(f"number of iterations completed: {regression.n_iter_}, number of weight updates: {regression.t_}")

w_coefficients = regression.coef_
b_coefficient = regression.intercept_


print("Final equation is:\nF(x) = ", end="")
for i in range(w_coefficients.shape[0]):
    print(f"({w_coefficients[i]})X{i+1} + ", end="")
print(b_coefficient[0])

print(f"where X1 is Size_(sqft), X2 is Number_of_Bedrooms, X3 is Number_of_floors, X4 is Age_of_Home,"
      f"b is base price and "
      f"output is price in 1000s $")


def get_array_of_predictions_y(x_lists, w_array, b):
    """returns an array of results y from a linear regression model, given wb coefficients and x data"""
    y_array = np.dot(x_lists, w_array) + b

    return y_array



# plotting each feature with y output (training vs predictions)

prediction_array = get_array_of_predictions_y(x_lists=x_normalised,
                                             w_array=w_coefficients,
                                             b=b_coefficient)

peak_numbers = np.ptp(x_data_arrays, axis=0)
print(peak_numbers)


fig, ax = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
for i in range(len(ax)):
    current_feature_array = np.array([list[i] for list in x_normalised])
    ax[i].scatter(current_feature_array, y_data_array, label="Actual data")
    ax[i].set_xlabel(feature_names[i], fontsize=15)
    ax[i].scatter(current_feature_array, prediction_array, c='r', label="Prediction data")
ax[0].set_ylabel("Price", fontsize=15)
ax[0].legend()
fig.suptitle("Comparison of Actual and prediction data for multi-feature linear regression model (using Scikit learn)")

plt.show()


max_size = features_dataframe.max()["Size_(sqft)"]
max_bedrooms = features_dataframe.max()["Number_of_Bedrooms"]
max_floors = features_dataframe.max()["Number_of_floors"]
max_age = features_dataframe.max()["Age_of_Home"]
max_price = features_dataframe.max()["Price_(1000s_dollars)"]

variables = [int(input("Enter size in sqft: "))/max_size, int(input("Number of bedrooms: "))/max_bedrooms,
             int(input("Number of floors: "))/max_floors,
             int(input("Age: "))/max_age]

variables = np.array(variables)
price = np.dot(variables, w_coefficients) + b_coefficient

print(price*max_price)
-