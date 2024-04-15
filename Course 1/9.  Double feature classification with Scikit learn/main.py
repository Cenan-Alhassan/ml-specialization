import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data
from sklearn.linear_model import LogisticRegression

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])



# z = w1x1 + w2x2 + b
# we would like to plot the decision boundary. We require 2 w coefficients and a b coefficient

# create the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
print(f"number of iterations completed: {lr_model.n_iter_}")

w = lr_model.coef_
b = lr_model.intercept_
w = w.flatten()
print(f"w1: {w[0]}, w2: {w[1]}, b: {b}")

fig , ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

input1, input2 = (float(input("input x1: ")), float(input("input x2: ")))
prediction = lr_model.predict(np.array([input1, input2]).reshape(1, -11))

print(f"The model predicts the output to be: {'0' if prediction[0] == 0 else 'X'}", end="")

ax.axis([0, 6, 0, 6])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
x0 = np.arange(0, 3.5, 0.1)
# after division:
x1 = -1.22862 * x0 + 3.16629
ax.plot(x0, x1, lw=1, label="decision boundary")
plt.legend()
plt.show()



