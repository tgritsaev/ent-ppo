"""Check the actual Ising TRPO step on a small masked categorical policy."""

import ast
from pathlib import Path
from types import SimpleNamespace
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import gfnx
from jax.lax import stop_gradient

source = Path(__file__).resolve().parents[1] / "baselines/TRPO_ising_bitseq.py"
tree = ast.parse(source.read_text())
helpers = [
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and (n.name.startswith("tree_") or n.name == "conjugate_gradients")
]
train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_step")
a = next(
    i
    for i, n in enumerate(train.body)
    if isinstance(n, ast.FunctionDef) and n.name == "forward_log_probs_from_params"
)
b = next(
    i
    for i, n in enumerate(train.body)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "metrics_state" for t in n.targets)
)


class Policy(eqx.Module):
    logits: jax.Array

    def __call__(self, obs):
        return {"forward_logits": self.logits[obs[0]]}


class Env:
    def get_invalid_mask(self, state, params):
        return state


policy = Policy(jnp.array([[0.1, -0.3, 0.0], [0.2, 0.1, -0.4]]))
obs = jnp.array([[[0], [1], [0]], [[1], [0], [0]]])
pad = jnp.array([[False, False, True], [False, False, True]])
invalid = jnp.array(
    [
        [[False, False, True], [False, False, False], [False, False, True]],
        [[False, False, False], [False, False, True], [False, False, True]],
    ]
)
actions = jnp.array([[0, 1, 0], [2, 1, 0]])
old = jax.nn.log_softmax(gfnx.utils.mask_logits(policy.logits[obs[..., 0]], invalid), axis=-1)[
    :, :-1
]
sampled = jnp.take_along_axis(old, actions[:, :-1, None], axis=-1)[..., 0]
aux = {
    "advantages_old": jnp.array([[1.0, -0.5], [-0.8, 0.7]]),
    "log_pf_old_full": old,
    "log_pf_old_sampled": sampled,
    "pad_mask": pad[:, :-1],
}
scope = dict(
    jax=jax,
    jnp=jnp,
    eqx=eqx,
    gfnx=gfnx,
    policy_params=policy,
    policy_static=eqx.filter(policy, eqx.is_array, inverse=True),
    env=Env(),
    env_params=None,
    traj_data=SimpleNamespace(obs=obs, state=invalid, action=actions, pad=pad),
    trpo_aux_info=aux,
    valid=~pad[:, :-1],
    pad_mask=pad[:, :-1],
    n_valid=4,
    cg_damping=0.01,
    cg_iters=10,
    trpo_delta=0.01,
    line_search_iters=10,
    line_search_shrink=0.5,
)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=helpers + train.body[a:b], type_ignores=[])),
        str(source),
        "exec",
    ),
    scope,
)
assert bool(scope["trpo_accepted"])
assert float(scope["final_policy_loss"]) < float(scope["policy_loss"])
assert -1e-6 <= float(scope["final_policy_kl"]) <= 0.01
print(
    "Actual TRPO update decreases surrogate and satisfies KL constraint:",
    float(scope["final_policy_kl"]),
)
# Check actual score extraction on non-autoregressive Ising trajectories.
env = gfnx.IsingBitseqEnvironment(
    gfnx.BitseqIsingRewardModule(L=2, k=2, beta=0.6), L=2, k=2, beta=0.6
)
params = env.init(jax.random.PRNGKey(1))


class Uniform:
    def __call__(self, obs):
        return {
            "forward_logits": jnp.zeros(env.action_space.n),
            "backward_logits": jnp.zeros(env.backward_action_space.n),
        }


model = Uniform()


def fwd(key, obs, unused):
    outputs = jax.vmap(model)(obs)
    return outputs["forward_logits"], outputs


traj, info = gfnx.utils.forward_rollout(jax.random.PRNGKey(2), 4, fwd, None, env, params)
extract = next(
    n
    for n in train.body
    if isinstance(n, ast.FunctionDef) and n.name == "extract_advantage_components"
)
scope.update(
    TransformerPolicy=Uniform,
    TransformerBaseline=Uniform,
    chex=SimpleNamespace(Array=jax.Array),
    stop_gradient=stop_gradient,
)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[extract], type_ignores=[])), str(source), "exec"
    ),
    scope,
)
c = scope["extract_advantage_components"](
    model, lambda obs: jnp.array(0.0), jnp.array(2.0), traj, env, params
)
pf, pb = gfnx.utils.forward_trajectory_log_probs(env, traj, params)
np.testing.assert_allclose(c["scores"].sum(1), pf - pb - info["log_gfn_reward"] + 2, atol=1e-6)
print("Actual Ising TRPO scores include every backward transition and terminal reward.")
