import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data


def linear_cost(x, y_actual, y_pred, w):
    global LAMBDA_R
    m = x.shape[0]
    n = w.shape[0]
    cost = 0
    regularise = 0
    for i in range(m):
        cost += (y_pred[i] - y_actual[i]) ** 2
    cost *= 1 / (2 * m)

    for i in range(n):
        regularise += w[i]**2
    regularise *= LAMBDA_R / (2 * m)

    return cost + regularise


def polynomial_output(xr, w):
    """outputs array of y = w1x + w2x^2 + w3x^3 + w4x^4 - 1 for every x[i]"""
    for i in range(xr.shape[0]):
        features[i] = np.array([xr[i], xr[i]**2, xr[i]**3, xr[i]**4]).flatten()

    output = np.zeros(5)
    for i in range(xr.shape[0]):
        output[i] = np.dot(w, features[i]) - 1

    return output


def linear_dj_dw_i(x_array, y_actual, y_pred, w_i):
    global LAMBDA_R
    m = x_array.shape[0]
    derivative = 0
    for i in range(m):
        derivative += (y_pred[i] - y_actual[i]) * x_array[i]
    derivative *= 1 / m

    return derivative + (LAMBDA_R * w_i) / m


def linear_gradient_descent(x_array, y_actual, w):
    """Returns array of improved w coefficients"""
    global ALPHA_R, ITERATIONS_R
    n = w.shape[0]
    for i in range(ITERATIONS_R):
        predictions = polynomial_output(x_array, w)
        print(f"w {i}: {w}")
        for j in range(n):
            print(f"dj_dw_{j}: {linear_dj_dw_i(x_array, y_actual, predictions, w[j])}")
            w[j] = w[j] - ALPHA_R * linear_dj_dw_i(x_array, y_actual, predictions, w[j])

        print(linear_cost(x_array, y_actual, predictions, w))

    return w


# when in doubt, reduce learning rate
ALPHA_R = 0.0005
LAMBDA_R = 1
ITERATIONS_R = 100
xr = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
yr = np.array([2, 5, 8, 11, 14])

# instead of just finding the cost of a better w for x1,
# we also add the regularization for the w of the extra polynomial

# standard equation would be y = x - 1
# augmented equation is y = x + 0.2x^2 + 0.5x^3 + 0.2x^4 - 1
# must limit the extra coefficients through regularized cost

# w = [1, 0.2, 0.5, 0.2]
# features[i] = [x[i], x[i]^2, x[i]^3, x[i]^4]
# output[i] = np.dot(w, features[i])

# features = np.array(np.array([value, value**2, value**3, value**4]) for value in xr)
w = np.array([2, 0.2, 0.5, 0.2])
print(f"w before regularization: {w}")
# our aim is to minimize w2, w3, w4

features = np.zeros(24).reshape(-1, 4)


polynomial_y = polynomial_output(xr, w)
w_new = linear_gradient_descent(xr, yr, w)
print(f"\n\nw after {ITERATIONS_R} gradient descent{w_new}")
regularized_y = polynomial_output(xr, w_new)
print(f"X train: {xr}\nY train: {yr}")
print(f"prediction output of y = 2x + 0.2x^2 + 0.5x^3 + 0.2x^4 - 1 before regularization: {polynomial_y}")
print(f"cost: {linear_cost(xr, yr, polynomial_y, w)}")
print(f"Y after regularization at w {w} is:{regularized_y}")
print(f"cost: {linear_cost(xr, yr, regularized_y, w)}")


figure1 = plt.figure(1)
plt.scatter(xr, yr)
plt.plot(xr, xr * 3 - 1, label="y = 3x - 1")
plt.plot(xr, polynomial_y, label="Before regularized gradient descent")
plt.plot(xr, regularized_y, label="After regularized gradient descent")
plt.xlim(0, 6)
plt.ylim(0, 16)
figure1.suptitle("Trying to regularize y = w1x + w2x^2 + w3x^3 + w4x^4 - 1")
plt.ylabel("Y")
plt.xlabel("X")

plt.legend()
# plt.show()


# Classification

def polynomial_output_c(x1, x2, w1, w2):
    """outputs array of y = w1x + w2x^2 + w3x^3 + w4x^4 - 1 for every x[i]"""
    m = x1.shape[0]
    n = w1.shape[0]
    features = np.zeros(m*n).reshape(-1, n)
    for i in range(m):
        features[i] = np.array([x1[i], x1[i]**2, x1[i]**3, x1[i]**4]).flatten()
    output = np.zeros(m)
    for i in range(m):
        output[i] = np.dot(w1, features[i]) + w2 * x2[i] + 3

    # same bur=t different way
    output_poly = np.zeros(m)
    for i in range(m):
        output_poly[i] = -1*x1[i] - 0.1 * x1[i]**2 - 0.1 * x1[i]**3 + 0.1 * x1[i]**4 - x2[i] + 3
    return output

def loss(z, y, w, b):
    print(f"z: {z}")
    g = 1 / (1 + np.exp(-z))
    print(f"g: {g}")
    loss = -y * np.log10(g) - (1 - y) * np.log10(1 - g)
    print(f"loss at {g}, {y}: {loss}")

    return loss

def logistic_cost(x, y, w, b):
    print(w)
    m = x.shape[0]
    cost = 0
    for i in range(m):
        z = w[0]*x[i][0] + w[1] * x[i][0]**2 + w[2] * x[i][0]**3 + w[3] * x[i][0]**4 + w[4] * x[i][1] + 3
        g = 1 / (1 + np.exp(-z))
        cost += -y[i]*np.log10(g) - (1 - y[i])*np.log10(1 - g)

    cost *= (1/m)

    regularise = 0
    n = w.shape[0]
    for i in range(n):
        regularise += w[i]**2
    regularise *= LAMBDA_C / (2 * m)

    return cost + regularise


def logistic_dj_dw(x_array, y_actual, z, w):
    """Returns array of dj_dw"""
    x_array = np.array([[list[0], list[0] ** 2, list[0] ** 3, list[0] ** 4, list[1]] for list in x_array])
    global LAMBDA_C
    m = x_array.shape[0]
    n = w.shape[0]
    dj_dw = np.zeros(n)
    for j in range(n):
        derivative = 0
        for i in range(m):
            g = 1 / (1 + np.exp(-z[i]))
            derivative += (g - y_actual[i]) * x_array[i][j]
        derivative *= 1 / m
        derivative += (LAMBDA_C * w[j]) / m
        dj_dw[j] = derivative

    return dj_dw


def logistic_gradient_descent(x_array, y_actual, w):
    """Returns array of improved w coefficients"""
    xl1 = np.array([list[0] for list in x_array]).reshape(-1, 1)
    xl2 = np.array([list[1] for list in x_array])
    global ALPHA_L, ITERATIONS_L
    n = w.shape[0]
    for i in range(ITERATIONS_L):
        print(f"w {i}: {w}")
        z = polynomial_output_c(xl1, xl2, w[:-1], w[-1])
        dj_dw = logistic_dj_dw(x_array, y_actual, z, w)
        for i in range(n):
            print(f"dj_dw_{i}: {dj_dw[i]}" )
        w = w - ALPHA_L * dj_dw

        print(f"cost: {logistic_cost(x_array, y_actual, w, 0)}")
    return w


LAMBDA_C = 1
ALPHA_L = 0.05
ITERATIONS_L = 1000
w_init = np.array([-1, -0.1, -0.1, 2, -1])

xl = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
yl = np.array([0, 0, 0, 1, 1, 1])
xl1 = np.array([list[0] for list in xl]).reshape(-1, 1)
xl2 = np.array([list[1] for list in xl])
z = polynomial_output_c(xl1, xl2, w_init[:-1], w_init[-1])
print(z)

# x2 = -x + -x^2 + -x^3 + x^4 + 3 --> 0 = -x + -x^2 + -x^3 + x^4 + -x2 + 3
# w = [-1, -1, -1, 1], b = 3
x0 = np.arange(0, 3.5, 0.1).reshape(-1, 1)
x2 = np.arange(0, 3.5, 0.1)
x1 = -1.2 * x0 + 3

w_new = logistic_gradient_descent(xl, yl, w_init)

fig , ax = plt.subplots(1,1,figsize=(6.3,4.8))
plot_data(xl, yl, ax)
ax.axis([0, 6, 0, 6])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
ax.plot(x0, x1, lw=1, label="decision boundary y = -1.2x + 3")
ax.plot(x0, polynomial_output_c(x0, x2, w_init[:-1], w_init[-1]) + x2, lw=1, label="decision boundary before regularization")
ax.plot(x0, polynomial_output_c(x0, x2, w_new[:-1], w_new[-1]) + x2, lw=1, label="decision boundary after regularization")
ax.set_title("Trying to regularize y = w1x1 + w2x1^2 + w3x1^3 + w4x1^4 + w5x2 + 3", fontsize=10)
plt.legend()
plt.show()

