import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read csv file
df = pd.read_csv('HW2_data.csv', header=None, names=['x', 'y'])

# extract coordinates from DataFrame
x = df['x'].values
y = df['y'].values

# draw scatter plot
plt.scatter(x, y, c='blue', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of Ellipse Points')
plt.grid(True)

# show plot
plt.show()

# build matrices
N = len(x)
A = np.zeros((N, 2))
A[:, 0] = x ** 2
A[:, 1] = y ** 2
b = np.ones(N)


def gradient_descent(A, b, tol=1e-6, max_iter=1000):
    N, d = A.shape
    a = np.zeros(d)
    # step size
    mu = 1 / (2 * np.linalg.norm(np.dot(A.T, A), 2))

    for i in range(max_iter):
        f_prev = np.sum((np.dot(A, a) - b)**2)

        # compute gradient
        grad_f = 2 * np.dot(A.T, np.dot(A, a) - b)

        # update a
        a = a - mu * grad_f

        # compute new value of f(a)
        f_new = np.sum((np.dot(A, a) - b)**2)

        # check for convergence
        if np.abs(f_new - f_prev) < tol:
            print(f"Converged after {i+1} iterations.")
            return a

    print("Max iterations reached.")
    return a


# run gradient descent
a_optimal = gradient_descent(A, b)

# compute f(a)
f_a = np.sum((np.dot(A, a_optimal) - b)**2)

print(f"Optimal a: {a_optimal}")
print(f"f(a): {f_a}")
