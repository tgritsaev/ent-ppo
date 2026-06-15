import pytest
import torch
import torch_geometric.data as gd

from gflownet.algo.config import TBVariant
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.config import Config
from gflownet.trainer import GFNTrainer


class DummyContext:
    def has_n(self):
        return False


class FakeActionCategorical:
    def __init__(self, log_probs):
        self.log_probs = log_probs

    def log_prob(self, actions, logprobs=None, batch=None):
        return self.log_probs.to(actions.device)


class FakeModel(torch.nn.Module):
    def __init__(self, log_p_f):
        super().__init__()
        self.log_p_f = log_p_f

    def forward(self, batch, cond_info, batched=False):
        per_graph_out = torch.zeros((batch.x.shape[0], 1), device=batch.x.device)
        return FakeActionCategorical(self.log_p_f), per_graph_out

    def logZ(self, cond_info):
        return torch.zeros((cond_info.shape[0], 1), device=cond_info.device)


def make_algo(variant, do_sample_p_b=True):
    cfg = Config()
    cfg.algo.tb.variant = variant
    cfg.algo.tb.do_sample_p_b = do_sample_p_b

    algo = TrajectoryBalance.__new__(TrajectoryBalance)
    algo.ctx = DummyContext()
    algo.global_cfg = cfg
    algo.cfg = cfg.algo.tb
    algo.length_normalize_losses = False
    algo.reward_normalize_losses = False
    algo.reward_loss = cfg.algo.tb.loss_fn
    algo.tb_loss = cfg.algo.tb.loss_fn
    algo.mask_invalid_rewards = False
    algo.model_is_autoregressive = False
    return algo


def make_batch(elbo_mask=None, proxy_eubo_mask=None):
    batch = gd.Batch()
    batch.x = torch.ones((4, 1))
    batch.traj_lens = torch.tensor([2, 2])
    batch.log_rewards = torch.tensor([1.0, 2.0])
    batch.log_p_B = torch.tensor([0.1, 0.2, 0.3, 0.4])
    batch.actions = torch.zeros((4, 3), dtype=torch.long)
    batch.cond_info = torch.zeros((2, 1))
    batch.is_valid = torch.ones(2)
    batch.num_online = 1
    batch.num_offline = 1
    batch.elbo_mask = torch.tensor(elbo_mask or [True, False])
    batch.proxy_eubo_mask = torch.tensor(proxy_eubo_mask or [False, True])
    return batch


@pytest.mark.parametrize("variant", [TBVariant.TB, TBVariant.DB, TBVariant.SubTB1])
def test_elbo_and_proxy_eubo_use_shared_bound_formula_for_tb_variants(variant):
    algo = make_algo(variant)
    batch = make_batch()
    model = FakeModel(torch.tensor([0.5, 0.25, 0.75, 0.0]))

    _, info = algo.compute_batch_losses(model, batch)

    expected_bound = torch.tensor([1.0 + 0.1 + 0.2 - 0.5 - 0.25, 2.0 + 0.3 + 0.4 - 0.75 - 0.0])
    assert torch.isclose(info["elbo"], expected_bound[0])
    assert torch.isclose(info["proxy_eubo"], expected_bound[1])


def test_proxy_eubo_is_not_reported_when_backward_sampling_is_disabled():
    algo = make_algo(TBVariant.TB, do_sample_p_b=False)
    batch = make_batch(elbo_mask=[False, False], proxy_eubo_mask=[True, True])
    model = FakeModel(torch.tensor([0.5, 0.25, 0.75, 0.0]))

    _, info = algo.compute_batch_losses(model, batch)

    assert "elbo" not in info
    assert "proxy_eubo" not in info


def test_metrics_are_not_reported_without_explicit_trajectory_masks():
    algo = make_algo(TBVariant.TB)
    batch = make_batch(elbo_mask=[False, False], proxy_eubo_mask=[False, False])
    model = FakeModel(torch.tensor([0.5, 0.25, 0.75, 0.0]))

    _, info = algo.compute_batch_losses(model, batch)

    assert "elbo" not in info
    assert "proxy_eubo" not in info


def test_training_loader_rejects_dataset_trajectories():
    trainer = GFNTrainer.__new__(GFNTrainer)
    trainer.cfg = Config()
    trainer.cfg.algo.num_from_dataset = 1
    trainer.cfg.replay.use = False
    trainer.replay_buffer = None
    trainer.sampling_model = object()
    trainer._wrap_for_mp = lambda x: x

    with pytest.raises(AssertionError, match="proxy-EUBO validation"):
        trainer.build_training_data_loader()
