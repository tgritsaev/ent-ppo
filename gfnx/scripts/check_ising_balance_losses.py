import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import numpy as np
import jax
import jax.numpy as jnp
from ising_balance_losses import balance_loss

rng = np.random.default_rng(4)
pf = rng.normal(size=(3, 4))
pb = rng.normal(size=(3, 4))
flows = rng.normal(size=(3, 5))
pad = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], bool)
done = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]], bool)
rew = np.zeros((3, 4))
rew[0, 3] = 2
rew[1, 2] = 3
rew[2, 1] = 4
inc = np.where(pad, 0, pf - pb)
delta = inc.sum(1) - rew.sum(1)
flow = flows.copy()
flow[:, 1:] = np.where(done & ~pad, rew, flow[:, 1:])
z = 0.8
refs = {"tb": np.mean((z + delta) ** 2), "vargrad": 0.5 * np.var(delta, ddof=1)}
errs = []
sub = []
for n in range(3):
    T = int((~pad[n]).sum())
    weighted = []
    weights = []
    for i in range(T):
        errs.append(0.5 * (flow[n, i] + inc[n, i] - flow[n, i + 1]) ** 2)
        for j in range(i + 1, T + 1):
            w = 1.9 ** (j - i)
            weights.append(w)
            weighted.append(w * (flow[n, i] + inc[n, i:j].sum() - flow[n, j]) ** 2)
    sub.append(sum(weighted) / sum(weights))
refs.update(db=np.mean(errs), subtb=np.mean(sub))
for obj, ref in refs.items():
    args = list(map(jnp.asarray, [pf, pb, flows, rew, done, pad]))
    f = lambda x: balance_loss(obj, x, *args[1:], z)
    val = float(f(args[0]))
    np.testing.assert_allclose(val, ref, rtol=2e-6)
    grad = np.asarray(jax.grad(f)(args[0]))
    assert np.all(np.isfinite(grad))
    assert np.all(grad[pad] == 0)
    eps = 1e-2
    a = pf.copy()
    b = pf.copy()
    a[0, 0] += eps
    b[0, 0] -= eps
    np.testing.assert_allclose(
        grad[0, 0],
        (float(f(jnp.asarray(a))) - float(f(jnp.asarray(b)))) / (2 * eps),
        rtol=0.01,
        atol=1e-4,
    )
    print(obj, "reference and gradient checks passed:", val)
# Any perfectly balanced trajectory makes all four losses zero.
pf = jnp.zeros((3, 4))
pb = jnp.zeros_like(pf)
flows = jnp.ones((3, 5)) * 2
rew = jnp.zeros_like(pf).at[:, -1].set(2)
done = jnp.zeros_like(pf, dtype=bool).at[:, -1].set(True)
pad = jnp.zeros_like(done)
for obj in refs:
    np.testing.assert_allclose(balance_loss(obj, pf, pb, flows, rew, done, pad, 2), 0)
print("All zero-residual checks passed")
