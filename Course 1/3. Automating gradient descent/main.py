# My first training model!
# linear regression model using gradient descent for training


import numpy as np
import matplotlib.pyplot as plt


def output_function_of_wb(x_dataset, w, b):
    """taking a certain w and b for the function f_wb(x) = w*x + b, returns np array of outputs y given an np array
    of input data x"""

    dimension = x_dataset.shape[0]

    f_wb = np.zeros(dimension)

    for i in range(dimension):
        f_wb[i] = w * x_dataset[i] + b

    return f_wb


def cost_function(x_dataset, y_dataset, w, b):
    """Checks the error between the original dataset and the output of the function with chosen w, b by checking
    difference between each real output of x, and the predicted output of the f_wb(x) of the same x"""

    dimension = x_dataset.shape[0]
    cost = 0
    for i in range(dimension):
        cost += ((w * x_dataset[i] + b) - y_dataset[i]) ** 2

    cost *= 1/(2*dimension)

    return cost

def cost_derivative_for_w(x_dataset, y_dataset, w, b):
    """Takes the x data, y data, w and b coefficients and returns a tuple of the derivative of the cost function
    with respect w as the first value, and with respect to b as the second value"""

    dimension = x_dataset.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(dimension):
        dj_dw += ((w * x_dataset[i] + b) - y_dataset[i]) * x_dataset[i]
        dj_db += ((w * x_dataset[i] + b) - y_dataset[i])

    dj_dw *= 1 / dimension
    dj_db *= 1 / dimension

    return dj_dw, dj_db


w = 430
b = 20
learning_rate = 0.3
x_data = np.array([1.0, 1.1, 1.5, 1.7,  2.0, 2.1, 2.5, 3.0])
y_data = np.array([300.0, 320.0, 400.0, 420.0, 500.0, 525.0, 550, 700.0])

# scatter the training data on graph
plt.scatter(x_data[0:], y_data[0:], marker='x', c='r', label="Actual Values")
plt.plot(x_data[0:], output_function_of_wb(x_data, w, b)[0:], c='g', label=f"Initial w={w}, b={b}")

plt.ylabel("Price (in 1000s of dollars")
plt.xlabel("Size (in 1000 square ft)")


# implement convergence algorithm
for i in range(200):
    temp_w = w - learning_rate * cost_derivative_for_w(x_data, y_data, w, b)[0]
    temp_b = b - learning_rate * cost_derivative_for_w(x_data, y_data, w, b)[1]

    w = temp_w
    b = temp_b

    print(f"{i}- w = {w:0.2f}, b = {b:0.2f}\n"
          f"cost = {cost_function(x_data, y_data, w, b)} \n")
    plt.plot(x_data[0:], output_function_of_wb(x_data, w, b)[0:])


plt.plot(x_data[0:], output_function_of_wb(x_data, w, b)[0:], label=f"final convergence w={w:0.2f}"
                                                                              f", b={b:0.2f}")


# shows labels
plt.legend()
# shows all data
plt.show()
