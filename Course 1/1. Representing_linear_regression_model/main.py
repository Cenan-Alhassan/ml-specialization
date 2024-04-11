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


x_data = np.array([1.0, 1.5, 2.0])
y_data = np.array([300.0, 400.0, 500.0])

# scatter the training data on graph
plt.scatter(x_data[0:], y_data[0:], marker='x', c='r', label="Actual Values")


# for i in range(1, 4):
#     line = output_function_of_wb(x_data, 100*i, 300-i*100)
#     plt.plot(x_data[0:], line[0:])


plt.ylabel("Price (in 1000s of dollars")
plt.xlabel("Size (in 1000 square ft)")


data_predictions = output_function_of_wb(x_dataset=x_data, w=200, b=100)
plt.plot(x_data[0:], data_predictions[0:], label='predicted values')


make_prediction = float(input("Enter size of house: "))
# a size (x value), is given.
# make a prediction by substituting in the regression model f_wb(x)
# in this program, we found the model by manually selecting w and b as the most suitable coefficients for our data
prediction_output = 200 * make_prediction + 100
plt.scatter(make_prediction, prediction_output, c='g', label=f"prediction = {prediction_output}")

# shows labels
plt.legend()
# shows all data
plt.show()
