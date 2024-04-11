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

features_dataframe = pd.read_csv("House_features.csv")
feature_names = [name for name in features_dataframe.columns]

# scale all features before converting to lists


try:
    features_dataframe["income"]

except KeyError:

    max_size = features_dataframe.max()["Size_(sqft)"]
    max_bedrooms = features_dataframe.max()["Number_of_Bedrooms"]
    max_floors = features_dataframe.max()["Number_of_floors"]
    max_age = features_dataframe.max()["Age_of_Home"]
    max_price = features_dataframe.max()["Price_(1000s_dollars)"]

    for i, j in features_dataframe.iterrows():
        j.Number_of_Bedrooms = j.Number_of_Bedrooms / max_bedrooms
        j["Size_(sqft)"] = round(j["Size_(sqft)"] / max_size, 3)
        j.Number_of_floors = j.Number_of_floors / max_floors
        j.Age_of_Home = round(j.Age_of_Home / max_age, 3)
        j["Price_(1000s_dollars)"] = round(j["Price_(1000s_dollars)"] / max_price, 3)


else:
    max_age = features_dataframe.max()["age"]
    max_experience = features_dataframe.max()["experience"]
    max_income = features_dataframe.max()["income"]

    print(max_age, max_experience, max_income)

    for i, j in features_dataframe.iterrows():
        j.age = round(j.age / max_age)
        j.experience = round(j.experience / max_experience)
        j.income = round(j.income / max_income)

        features_dataframe.loc[i] = j  # if don't include this, j will not be updated to dataframe


dataframe_lists = features_dataframe.values.tolist()

# ith row of x values from the csv data would be dataframe_lists[i][:-1]

number_of_features = len(dataframe_lists[0][:-1])
w_coefficients = np.ones(number_of_features)
b_coefficient = 3
learning_rate = 0.5

y_data_list = [float(list[-1]) for list in dataframe_lists[0:]]

# x data list holds integers now
x_data_lists = [list[:-1] for list in dataframe_lists[0:]]

print(dataframe_lists)


def f_vw_b(x_features_array, vw, b):
    f_x = np.dot(x_features_array, vw) + b
    return f_x


def cost_function_vw_b(y_data, x_data, vw, b):
    cost = 0

    dimension = np.array(y_data).shape[0]
    for i in range(dimension):
        current_x_features = [float(value) for value in x_data[i]]
        x_array = np.array(current_x_features)
        function_output = (np.dot(x_array, vw) + b)
        cost += (function_output - y_data[i]) ** 2

    cost *= 1 / (2 * dimension)

    return cost


# create a list of a cost derivative version of w
def cost_function_derivative_for_wi(iteration, y_data, x_data, vw, b):
    derivative = 0

    dimension = np.array(y_data).shape[0]
    for i in range(dimension):
        current_x_features = [float(value) for value in x_data[i]]
        x_array = np.array(current_x_features)
        function_output = (np.dot(x_array, vw) + b)
        derivative += (function_output - y_data[i]) * x_array[iteration]

    derivative *= 1 / dimension

    return derivative


def cost_function_derivative_for_b(y_data, x_data, vw, b):
    derivative = 0

    dimension = np.array(y_data).shape[0]
    for i in range(dimension):
        current_x_features = [float(value) for value in x_data[i]]
        x_array = np.array(current_x_features)
        function_output = (np.dot(x_array, vw) + b)
        derivative += (function_output - y_data[i])

    derivative *= 1 / dimension

    return derivative


# train the model
cost_history = []
for j in range(1000):
    w_derivatives = []
    for i in range(w_coefficients.shape[0]):
        w_derivative = cost_function_derivative_for_wi(iteration=i,
                                                       y_data=y_data_list,
                                                       x_data=x_data_lists,
                                                       vw=w_coefficients,
                                                       b=b_coefficient)

        w_derivatives.append(w_derivative)
    w_derivatives = np.array(w_derivatives)

    b_coefficient -= learning_rate * cost_function_derivative_for_b(y_data_list,
                                                                    x_data_lists,
                                                                    w_coefficients,
                                                                    b_coefficient)

    w_coefficients = np.subtract(w_coefficients, learning_rate * w_derivatives)

    cost_value = cost_function_vw_b(x_data=x_data_lists,
                                    y_data=y_data_list,
                                    vw=w_coefficients,
                                    b=b_coefficient)

    print(f"iteration {j}: {cost_value}")
    cost_history.append(cost_value)


print("Final equation is:\nF(x) = ", end="")

for i in range(w_coefficients.shape[0]):
    print(f"({w_coefficients[i]})X{i+1} + ", end="")

print(b_coefficient)

# age = int(input("Enter age: "))/58
# experience = int(input("Years of experience: "))/17
#
# salary = age * w_coefficients[0] + experience * w_coefficients[1] + b_coefficient
# print(salary*6.36)

print(f"where X1 is Size_(sqft), X2 is Number_of_Bedrooms, X3 is Number_of_floors, X4 is Age_of_Home,"
      f"b is base price and "
      f"output is price in 1000s $")

print(w_coefficients, b_coefficient)


def get_array_of_predictions_y(x_lists, w_array, b):
    """returns an array of results y from a linear regression model, given wb coefficients and x data"""
    dimension = len(x_lists)
    y_array = np.zeros(dimension)

    for i in range(dimension):
        feature_array = x_lists[i]
        y_array[i] = np.dot(feature_array, w_array) + b

    return y_array



# plotting each feature with y output (training vs predictions)
y_data_array = np.array(y_data_list)
prediction_array = get_array_of_predictions_y(x_lists=x_data_lists,
                                             w_array=w_coefficients,
                                             b=b_coefficient)

f1 = plt.figure(1)
iteration_cost = plt.plot(np.arange(0, 1000)[0:], np.array(cost_history)[0:])

fig, ax = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
for i in range(len(ax)):
    current_feature_array = np.array([list[i] for list in x_data_lists])
    ax[i].scatter(current_feature_array, y_data_array, label="Actual data")
    ax[i].set_xlabel(feature_names[i], fontsize=15)
    ax[i].scatter(current_feature_array, prediction_array, c='r', label="Prediction data")
ax[0].set_ylabel("Price", fontsize=15)
ax[0].legend()
fig.suptitle("Comparison of Actual and prediction data for multi-feature linear regression model")

plt.show()




variables = [int(input("Enter size in sqft: "))/2104, int(input("Number of bedrooms: "))/5,
             int(input("Number of floors: "))/2,
             int(input("Age: "))/45]

variables = np.array(variables)
price = np.dot(variables, w_coefficients) + b_coefficient

print(price*460)