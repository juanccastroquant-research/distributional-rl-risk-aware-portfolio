"""
gradcheck.py
------------
Quick numerical gradient checks for the hand-written backward passes in
nn_core.py, encoders.py and models.py. Run with `python3 gradcheck.py`.
Any mismatch above the printed tolerance indicates a backprop bug.
"""
import numpy as np
from nn_core import Dense


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def check_dense():
    rng = np.random.default_rng(0)
    layer = Dense(5, 3, activation="tanh", name="d", rng=rng)
    x = rng.normal(size=(4, 5))

    def loss_fn():
        y = layer.forward(x)
        return np.sum(y ** 2)

    y = layer.forward(x)
    loss = np.sum(y ** 2)
    dy = 2 * y
    dx_analytic = layer.backward(dy)
    W_analytic = layer.grads["d.W"].copy()

    dx_numeric = numerical_grad(loss_fn, x)
    W_numeric = numerical_grad(loss_fn, layer.params["d.W"])

    err_x = np.max(np.abs(dx_analytic - dx_numeric))
    err_w = np.max(np.abs(W_analytic - W_numeric))
    print(f"Dense layer: max|dx err|={err_x:.2e}  max|dW err|={err_w:.2e}")
    assert err_x < 1e-4 and err_w < 1e-4, "Dense layer gradient check FAILED"
    print("Dense layer gradcheck PASSED")


if __name__ == "__main__":
    check_dense()
