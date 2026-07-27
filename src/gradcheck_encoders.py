import numpy as np
from encoders import AttentionEncoder, GraphConvLayer
from gradcheck import numerical_grad


def check_attention():
    rng = np.random.default_rng(1)
    n, L, F, dk, dm = 4, 6, 5, 7, 8
    enc = AttentionEncoder(F, dk, dm, rng=rng)
    X = rng.normal(size=(n, L, F))

    def loss_fn():
        H = enc.forward(X)
        return np.sum(H ** 2)

    H = enc.forward(X)
    loss = np.sum(H ** 2)
    dH = 2 * H
    dX_analytic = enc.backward(dH)

    dX_numeric = numerical_grad(loss_fn, X)
    dWk_numeric = numerical_grad(loss_fn, enc.params["attn.Wk"])
    dWv_numeric = numerical_grad(loss_fn, enc.params["attn.Wv"])
    dv_numeric = numerical_grad(loss_fn, enc.params["attn.v"])

    err_x = np.max(np.abs(dX_analytic - dX_numeric))
    err_wk = np.max(np.abs(enc.grads["attn.Wk"] - dWk_numeric))
    err_wv = np.max(np.abs(enc.grads["attn.Wv"] - dWv_numeric))
    err_v = np.max(np.abs(enc.grads["attn.v"] - dv_numeric))
    print(f"AttentionEncoder: dX={err_x:.2e} dWk={err_wk:.2e} dWv={err_wv:.2e} dv={err_v:.2e}")
    assert max(err_x, err_wk, err_wv, err_v) < 1e-4, "AttentionEncoder gradcheck FAILED"
    print("AttentionEncoder gradcheck PASSED")


def check_gnn():
    rng = np.random.default_rng(2)
    n, d_in, d_out = 5, 6, 4
    layer = GraphConvLayer(d_in, d_out, rng=rng)
    H = rng.normal(size=(n, d_in))
    A = rng.uniform(size=(n, n))
    A_hat = A / A.sum(axis=1, keepdims=True)  # row-normalised, fixed (not a param)

    def loss_fn():
        out = layer.forward(H, A_hat)
        return np.sum(out ** 2)

    out = layer.forward(H, A_hat)
    dout = 2 * out
    dH_analytic = layer.backward(dout)

    dH_numeric = numerical_grad(loss_fn, H)
    dW_numeric = numerical_grad(loss_fn, layer.params["gnn.W"])

    err_h = np.max(np.abs(dH_analytic - dH_numeric))
    err_w = np.max(np.abs(layer.grads["gnn.W"] - dW_numeric))
    print(f"GraphConvLayer: dH={err_h:.2e} dW={err_w:.2e}")
    assert max(err_h, err_w) < 1e-4, "GraphConvLayer gradcheck FAILED"
    print("GraphConvLayer gradcheck PASSED")


if __name__ == "__main__":
    check_attention()
    check_gnn()
