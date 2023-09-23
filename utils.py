import numpy as np
from mindiff import ts


def gradient_check(X, y, model, forward, epsilon=1e-7):
    grad = X.grad
    gradapprox = np.zeros_like(X.data)

    it = np.nditer(X.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        x_plus_epsilon = X.data.copy()
        x_minus_epsilon = X.data.copy()
        x_plus_epsilon[ix] += epsilon
        x_minus_epsilon[ix] -= epsilon

        L_plus = forward(ts.Tensor(x_plus_epsilon), y, model)
        L_minus = forward(ts.Tensor(x_minus_epsilon), y, model)
        gradapprox[ix] = (L_plus.data - L_minus.data) / (2 * epsilon)

        it.iternext()

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    return difference, grad, gradapprox
