# with age and years of experience, predict income (multiply by 10,000). Dataset from Kaggle

import csv
import numpy as np

# csv data has been randomized by default
with open("multiple_linear_regression_dataset.csv") as csv_data:
    csv = csv.reader(csv_data)
    csv_rows = []

    for row in csv:
        csv_rows.append(row)

# ith row of x values from the csv data would be csv_rows[i][:-1]

number_of_features = len(csv_rows[0][:-1])
w_coefficients = np.ones(number_of_features)
b_coefficient = 3
learning_rate = 0.00113

y_data_list = [float(list[-1]) for list in csv_rows[1:]]

# x data list holds strings
x_data_lists = [list[:-1] for list in csv_rows[1:]]

print(y_data_list, x_data_lists)


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


print("Final equation is:\nF(x) = ", end="")

for i in range(w_coefficients.shape[0]):
    print(f"({w_coefficients[i]})X{i+1} + ", end="")

print(f"{b_coefficient}\n where X1 is age of employee, X2 is years of experience, b is base salary and "
      f"output is salary in 10,000s $")


age = int(input("Enter age: "))
experience = int(input("Years of experience: "))

salary = age * w_coefficients[0] + experience * w_coefficients[1] + b_coefficient
print(salary)