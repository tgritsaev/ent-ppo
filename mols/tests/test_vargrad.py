import pytest
import torch

from gflownet.algo.config import TBVariant
from gflownet.algo.trajectory_balance import TrajectoryBalance, batch_log_partition_variance
from gflownet.config import Config
from tests.test_trajectory_metrics import DummyContext, FakeModel, make_algo, make_batch


class NoLogZModel(FakeModel):
    def logZ(self, cond_info):
        raise AssertionError("VarGrad must not use a learned logZ")


def test_vargrad_matches_batch_optimal_tb_and_policy_gradient():
    log_pf = torch.tensor([0.5, 0.25, 0.75, 0.0], requires_grad=True)
    loss, info = make_algo(TBVariant.VarGrad).compute_batch_losses(NoLogZModel(log_pf), make_batch())
    # z = [0.55, 1.95], mean=1.25, deviations=[-0.7, 0.7].
    assert loss.item() == pytest.approx(0.49)
    assert info['logZ'].item() == pytest.approx(1.25)
    loss.backward()
    torch.testing.assert_close(log_pf.grad, torch.tensor([0.7, 0.7, -0.7, -0.7]))


def test_vargrad_is_invariant_to_constant_reward_rescaling():
    algo = make_algo(TBVariant.VarGrad)
    model = NoLogZModel(torch.tensor([0.5, 0.25, 0.75, 0.0]))
    batch = make_batch()
    first, first_info = algo.compute_batch_losses(model, batch)
    batch.log_rewards += 5
    shifted, shifted_info = algo.compute_batch_losses(model, batch)
    torch.testing.assert_close(first, shifted)
    assert (shifted_info['logZ'] - first_info['logZ']).item() == pytest.approx(5)


def test_vargrad_recomputes_logz_for_reused_batch():
    algo = make_algo(TBVariant.VarGrad)
    batch = make_batch()
    pf = torch.tensor([0.5, 0.25, 0.75, 0.0])
    model = NoLogZModel(pf)
    _, first = algo.compute_batch_losses(model, batch)
    pf[:2] += 0.5
    _, second = algo.compute_batch_losses(model, batch)
    assert (second['logZ'] - first['logZ']).item() == pytest.approx(-0.5)


def test_vargrad_uses_tb_reward_clipping():
    algo = make_algo(TBVariant.VarGrad)
    batch = make_batch()
    batch.log_rewards[0] = -1000
    model = NoLogZModel(torch.tensor([0.5, 0.25, 0.75, 0.0]))
    loss, _ = algo.compute_batch_losses(model, batch)
    batch.log_rewards[0] = algo.global_cfg.algo.illegal_action_logreward
    clipped_loss, _ = algo.compute_batch_losses(model, batch)
    torch.testing.assert_close(loss, clipped_loss)


def test_vargrad_rejects_mixed_temperatures():
    cfg = Config()
    cfg.algo.tb.variant = TBVariant.VarGrad
    cfg.cond.temperature.sample_dist = 'uniform'
    with pytest.raises(ValueError, match='fixed reward temperature'):
        TrajectoryBalance(None, DummyContext(), cfg)


def test_population_variance_and_detached_intercept():
    z = torch.tensor([1., 3., 5.], requires_grad=True)
    losses, logz = batch_log_partition_variance(z)
    assert losses.mean().item() == pytest.approx(8 / 3)
    assert not logz.requires_grad
    losses.mean().backward()
    torch.testing.assert_close(z.grad, torch.tensor([-4/3, 0., 4/3]))
