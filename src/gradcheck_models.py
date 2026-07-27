import numpy as np
from models import DirichletActor, QuantileCritic, quantile_huber_loss_grad
from gradcheck import numerical_grad

CFG = dict(d_attn=6, d_model=5, d_graph=4, actor_hidden=10, critic_hidden=10,
           alpha_floor=1.0, n_quantiles=9)


def check_dirichlet_logprob_entropy():
    rng = np.random.default_rng(3)
    alpha = rng.uniform(0.5, 5.0, size=6)
    w = rng.dirichlet(alpha)

    def logp_fn():
        return DirichletActor.log_prob(w, alpha)

    analytic = DirichletActor.dlogprob_dalpha(w, alpha)
    numeric = numerical_grad(logp_fn, alpha)
    err = np.max(np.abs(analytic - numeric))
    print(f"Dirichlet log_prob d/dalpha: err={err:.2e}")
    assert err < 1e-4, "log_prob gradient FAILED"

    def ent_fn():
        return DirichletActor.entropy(alpha)

    analytic_h = DirichletActor.dentropy_dalpha(alpha)
    numeric_h = numerical_grad(ent_fn, alpha)
    err_h = np.max(np.abs(analytic_h - numeric_h))
    print(f"Dirichlet entropy d/dalpha: err={err_h:.2e}")
    assert err_h < 1e-4, "entropy gradient FAILED"
    print("Dirichlet log_prob/entropy gradchecks PASSED")


def check_actor_stack():
    rng = np.random.default_rng(4)
    n_assets, L, F = 5, 8, 4
    actor = DirichletActor(F, n_assets, CFG, rng=rng)
    X = rng.normal(size=(n_assets, L, F))
    A = rng.uniform(0, 1, size=(n_assets, n_assets))
    A_hat = A / A.sum(1, keepdims=True)
    prev_w = np.ones(n_assets) / n_assets

    def loss_fn():
        alpha = actor.forward(X, A_hat, prev_w)
        return np.sum(alpha ** 2)

    alpha = actor.forward(X, A_hat, prev_w)
    dalpha = 2 * alpha
    actor.zero_grad()
    actor.backward(dalpha)

    # check one weight matrix deep in the stack (attention Wv) receives a
    # sensible gradient matching finite differences
    analytic = actor.encoder.attn.grads["attn.Wv"].copy()
    numeric = numerical_grad(loss_fn, actor.encoder.attn.params["attn.Wv"])
    err = np.max(np.abs(analytic - numeric))
    print(f"Actor end-to-end (attn.Wv) grad err={err:.2e}")
    assert err < 1e-3, "Actor stack gradcheck FAILED"

    analytic_out = actor.out.grads["actor_out.W"].copy()
    numeric_out = numerical_grad(loss_fn, actor.out.params["actor_out.W"])
    err_out = np.max(np.abs(analytic_out - numeric_out))
    print(f"Actor end-to-end (actor_out.W) grad err={err_out:.2e}")
    assert err_out < 1e-3, "Actor head gradcheck FAILED"
    print("Actor end-to-end gradcheck PASSED")


def check_critic_stack():
    rng = np.random.default_rng(5)
    n_assets, L, F = 5, 8, 4
    critic = QuantileCritic(F, n_assets, CFG, rng=rng)
    X = rng.normal(size=(n_assets, L, F))
    A = rng.uniform(0, 1, size=(n_assets, n_assets))
    A_hat = A / A.sum(1, keepdims=True)
    prev_w = np.ones(n_assets) / n_assets
    action_w = rng.dirichlet(np.ones(n_assets))

    def loss_fn():
        q = critic.forward(X, A_hat, prev_w, action_w)
        return np.sum(q ** 2)

    q = critic.forward(X, A_hat, prev_w, action_w)
    dq = 2 * q
    critic.zero_grad()
    critic.backward(dq)

    analytic = critic.encoder.gnn.grads["gnn.W"].copy()
    numeric = numerical_grad(loss_fn, critic.encoder.gnn.params["gnn.W"])
    err = np.max(np.abs(analytic - numeric))
    print(f"Critic end-to-end (gnn.W) grad err={err:.2e}")
    assert err < 1e-3, "Critic stack gradcheck FAILED"
    print("Critic end-to-end gradcheck PASSED")


def check_quantile_huber():
    rng = np.random.default_rng(6)
    N = 9
    taus = (np.arange(N) + 0.5) / N
    pred = rng.normal(size=N)
    target = rng.normal(size=N + 2)

    def loss_fn():
        loss, _ = quantile_huber_loss_grad(pred, target, taus)
        return loss

    _, analytic = quantile_huber_loss_grad(pred, target, taus)
    numeric = numerical_grad(loss_fn, pred)
    err = np.max(np.abs(analytic - numeric))
    print(f"Quantile Huber loss grad err={err:.2e}")
    assert err < 1e-4, "Quantile Huber gradient FAILED"
    print("Quantile Huber gradcheck PASSED")


if __name__ == "__main__":
    check_dirichlet_logprob_entropy()
    check_actor_stack()
    check_critic_stack()
    check_quantile_huber()
