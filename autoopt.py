import matplotlib.pyplot as plt
import numpy as np
from rosenbrock import rosenbrock
from functools import partial
from autograd import jacobian, hessian
from parula import parula_map


# Automatic differentiation based newton optimizer

def optimize(objective, x0, tol):
    '''
    Optimizes the function `objective` (2D) using a Newton-based
    automatic differentiation method.

    objective: function to optimize
    x0: initial guess
    fd_step: size of step to use for finite difference
    tol: tolerance for convergence
    '''
    previous_step = 10
    n = len(x0)  # dimension of problem
    x = x0

    history = np.zeros((0, n))
    history = np.append(history, np.array([x]), axis=0)

    while abs(previous_step) > tol:
        # Compute gradient and Hessian using autograd
        J = jacobian(objective)(x)
        H = hessian(objective)(x)

        dx = np.linalg.solve(H, -J)
        x = x + dx
        history = np.append(history, np.array([x]), axis=0)
        previous_step = np.linalg.norm(dx)

    return x, history


if __name__ == "__main__":
    obj = partial(rosenbrock, alpha=10)

    x, history = optimize(obj, np.array([0., 1.]), 1e-6)

    xx, yy = np.meshgrid(np.linspace(-1.25, 1.25, 100), np.linspace(-0.75, 1.75, 100))

    # Draw contours with lines
    plt.contourf(xx, yy, obj([xx, yy]), 50, cmap=parula_map)
    plt.contour(xx, yy, obj([xx, yy]), 50, colors='k', linewidths=0.5)

    # Draw the history of the optimization with dots and lines
    plt.plot(history[:, 0], history[:, 1], 'ro-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
