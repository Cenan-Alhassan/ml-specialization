
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def loss(x, y, w, b):
    z = x * w + b
    print(f"z: {z}")
    g = 1 / (1 + np.exp(-z))
    print(f"g: {g}")
    loss = -y * np.log10(g) - (1 - y) * np.log10(1 - g)
    print(f"loss at {g}, {y}: {loss}")

    return loss

def cost_function(x, y, w, b):

    m = x.shape[0]
    cost = 0
    for i in range(m):
        z = x[i] * w + b
        g = 1 / (1 + np.exp(-z))
        cost += -y[i]*np.log10(g) - (1 - y[i])*np.log10(1 - g)

    cost *= (1/m)

    return cost

def gradient_descent_w(x, y, w, b, alpha):
    global B
    m = x.shape[0]
    iterations = 400
    history = np.zeros(iterations)
    for i in range(iterations):
        inner = 0
        for j in range(m):
            z = w*x[j] + b
            g = 1 / (1 + np.exp(-z))
            inner += (g - y[j]) * x[j]

        inner *= 1/m
        w = w - alpha * inner
        print(w)

        history[i] = cost_function(x, y, w, B)

    figure3 = plt.figure(3)
    plt.plot(np.arange(0, iterations, 1)[0:], history[0:], label=f"iterations at learning rate {LEARNING_RATE}")
    plt.legend()

LEARNING_RATE = 0.05
B = -21
x = np.array([1, 2, 3, 4, 5, 6])
# z_scalar = StandardScaler()
# x = z_scalar.fit_transform(x.reshape(-1, 1)) * 3
# x = x.flatten()
y = np.array([0, 0, 0, 1, 1, 1])
print(x, y)

gradient_descent_w(x, y, 1, B, LEARNING_RATE)


figure1 = plt.figure(1)
plt.scatter(x[0:], y[0:], marker='*')



dummy = np.arange(1, 6, 0.1)
dummy_z = 6*dummy - 21
plt.plot(dummy[0:], dummy_z, label=f"z = 6x - 21")
plt.ylim(-0.3, 1.3)
plt.plot(dummy[0:], 1 / (1 + np.exp(-dummy_z)), label="sigmoid of z")

# cost vs w graph (b is constant)
figure2 = plt.figure(2)
dummy2 = np.arange(1, 10, 0.1)
dummy_cost = np.array([cost_function(x, y, value, B) for value in dummy2])
plt.plot(dummy2[0:], dummy_cost[0:])
plt.xlabel("w of z = wx+b")
plt.ylabel("cost")



plt.legend()
plt.show()

print(cost_function(x, y, 6, B))





# map the cost of different g(z) according to w
