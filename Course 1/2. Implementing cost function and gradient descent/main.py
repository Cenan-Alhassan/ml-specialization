import numpy as np
import ipywidgets
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

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
    print(cost)


x_data = np.array([1.0, 1.5, 2.0])
y_data = np.array([300.0, 400.0, 500.0])

# scatter the training data on graph
plt.scatter(x_data[0:], y_data[0:], marker='x', c='r', label="Actual Values")



plt.ylabel("Price (in 1000s of dollars")
plt.xlabel("Size (in 1000 square ft)")

w = 199
b = 80

data_predictions = output_function_of_wb(x_dataset=x_data, w=w, b=b)
plt.plot(x_data[0:], data_predictions[0:], label='predicted values')

cost_function(x_data, y_data, w=w, b=b)


plt.close("all")
fig, ax, dyn_items = plt_stationary(x_data, y_data)
updater = plt_update_onclick(fig, ax, x_data, y_data, dyn_items)

# shows labels
plt.legend()
# shows all data
plt.show()
