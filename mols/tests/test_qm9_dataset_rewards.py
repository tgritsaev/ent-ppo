from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gflownet.data.qm9 import QM9Dataset
from gflownet.utils.eval_metrics import add_bound_metrics
from gflownet.utils.transforms import to_logreward


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / 'qm9.h5'
    pd.DataFrame({'SMILES': ['C', 'CC'], 'gap': [0.25, 0.3]}).to_hdf(path, key='df')
    data = QM9Dataset(path, ratio=1)
    data.idcs = np.array([0, 1])
    yield data
    data.terminate()


def test_dataset_uses_policy_reward_scorer_and_caches_untempered_rewards(dataset):
    calls = []
    def score(mols):
        calls.extend(mols)
        return torch.tensor([[0.875]]), torch.tensor([True])
    dataset.setup(SimpleNamespace(compute_obj_properties=score), SimpleNamespace(obj_to_graph=lambda m: m))
    _, reward = dataset[0]
    torch.testing.assert_close(reward, torch.tensor([0.875]))
    log_r = 16 * to_logreward(reward)
    assert log_r.item() == pytest.approx(16 * np.log(0.875))
    # The EUBO weights must be surrogate reward**beta, not raw gap**beta.
    info = {}
    add_bound_metrics(info, torch.tensor([3.]), log_r.reshape(1), torch.tensor([0.]), {'': torch.tensor([True])})
    assert info['iw_eubo_stat_weight_sum'].item() == pytest.approx(0.875**16)
    reward.fill_(0)
    torch.testing.assert_close(dataset[0][1], torch.tensor([0.875]))
    assert len(calls) == 1


def test_dataset_refuses_raw_gap_fallback(dataset):
    with pytest.raises(RuntimeError, match='setup'):
        dataset[0]


def test_dataset_marks_unscorable_molecule_for_omission(dataset):
    task = SimpleNamespace(compute_obj_properties=lambda mols: (torch.empty((0, 1)), torch.tensor([False])))
    dataset.setup(task, SimpleNamespace(obj_to_graph=lambda m: m))
    _, reward = dataset[0]
    assert not dataset.is_valid(0)
    torch.testing.assert_close(reward, torch.zeros(1))


def test_evaluation_omits_unscorable_before_backward_sampling(dataset):
    from gflownet.data.data_source import DataSource
    calls = []
    def score(mols):
        valid = mols[0].GetNumAtoms() == 1
        return (torch.tensor([[0.875]]) if valid else torch.empty((0, 1))), torch.tensor([valid])
    dataset.setup(SimpleNamespace(compute_obj_properties=score), SimpleNamespace(obj_to_graph=lambda m: m))
    source = DataSource.__new__(DataSource)
    source.iterators = []
    source.current_iter = 0
    source.is_algo_eval = True
    source.iterate_indices = lambda total, batch: [[1], [0]]  # Entirely invalid batch, then valid.
    source._timing = lambda *args: None
    source._timing_done = lambda *args: None
    def backwards(objs, model, encoding, p):
        calls.extend(objs)
        assert len(encoding) == len(objs) == 1
        return [{} for obj in objs]
    source.algo = SimpleNamespace(get_random_action_prob=lambda t: 0,
        create_training_data_from_graphs=backwards, cfg=SimpleNamespace(do_sample_p_b=True))
    source.task = SimpleNamespace(sample_conditional_information=lambda n, t: {'encoding': torch.zeros(n, 1)})
    source.compute_log_rewards = lambda trajs: None
    source.do_dataset_in_order(dataset, 1, None, num_total=2)
    batches = list(source.iterators[0]())
    assert len(calls) == len(batches) == 1
    assert calls[0].GetNumAtoms() == 1
    torch.testing.assert_close(batches[0][0][0]['obj_props'], torch.tensor([0.875]))
