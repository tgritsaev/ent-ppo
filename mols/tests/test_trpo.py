from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gflownet.algo.config import TRPOConfig, resolve_trpo_regime
from gflownet.algo.trpo import TRPO, backtracking, conjugate_gradients, flat_grad, gae, old_to_new_kl
from gflownet.config import Config


def categorical(logits):
    return SimpleNamespace(logsoftmax=lambda: [logits.log_softmax(-1)],
                           num_graphs=len(logits), batch=[torch.arange(len(logits))])


def test_exact_masked_kl_and_hvp():
    logits = torch.tensor([[0.2, -0.3, -torch.inf], [-1., 0., 1.]], dtype=torch.double, requires_grad=True)
    old = logits.detach().log_softmax(-1)
    kl = old_to_new_kl(categorical(logits), [old]).sum()
    g = flat_grad(kl, [logits], create_graph=True)
    v = torch.tensor([[0.3, -0.1, 0.], [0.1, 0.7, -0.2]], dtype=torch.double)
    hv = flat_grad((g * v.flatten()).sum(), [logits]).view_as(logits)
    p = old.exp()
    expected = p * (v - (p * v).sum(-1, keepdim=True))
    torch.testing.assert_close(hv, expected)
    def grad_at(x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(old_to_new_kl(categorical(x), [old]).sum(), x)[0]
    eps = 1e-5
    torch.testing.assert_close(hv, (grad_at(logits + eps*v)-grad_at(logits-eps*v))/(2*eps))
    new = logits.detach() + v
    got = old_to_new_kl(categorical(new), [old])
    expected_kl = (p * torch.where(torch.isfinite(old), old-new.log_softmax(-1), 0.)).sum(-1)
    torch.testing.assert_close(got, expected_kl)
    assert not torch.allclose(got, old_to_new_kl(categorical(logits), [new.log_softmax(-1)]))


def test_kl_rejects_changed_support():
    with pytest.raises(ValueError, match='supports'):
        old_to_new_kl(categorical(torch.tensor([[1., 2.]])), [torch.tensor([[0., -torch.inf]])])


def test_cg_matches_dense_solve_and_handles_zero_gradient():
    a = torch.tensor([[3., 1.], [1., 2.]], dtype=torch.double)
    b = torch.tensor([2., -1.], dtype=torch.double)
    torch.testing.assert_close(conjugate_gradients(lambda v: a @ v, b), torch.linalg.solve(a,b), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(conjugate_gradients(lambda v:a@v, torch.zeros_like(b)),torch.zeros_like(b))


def test_backtracking_accepts_only_improvement_within_kl():
    p = nn.Parameter(torch.tensor([1.]))
    def evaluate():
        return p.square().sum(), (p-1).square().sum()
    accepted, fraction, loss, kl = backtracking([p],torch.tensor([1.]),evaluate,torch.tensor(1.),0.1)
    assert accepted and fraction == 0.25
    assert loss < 1 and kl <= 0.1
    torch.testing.assert_close(p, torch.tensor([0.75]))


@pytest.mark.parametrize('raises',[False,True])
def test_failed_search_restores_parameters_exactly(raises):
    p = nn.Parameter(torch.tensor([0.321]))
    before=p.detach().clone()
    def evaluate():
        if raises: raise RuntimeError('candidate failed')
        return torch.tensor(2.), torch.tensor(0.)
    if raises:
        with pytest.raises(RuntimeError): backtracking([p],torch.ones(1),evaluate,torch.tensor(1.),0.01)
    else:
        assert not backtracking([p],torch.ones(1),evaluate,torch.tensor(1.),0.01)[0]
    assert torch.equal(p,before)


@pytest.mark.parametrize('regime,lam,critic_lam,epochs,splits',[('ours',.7,.7,4,8),('gfn_pg',.99,1.,1,1)])
def test_regime_targets_and_unscaled_centering(regime,lam,critic_lam,epochs,splits):
    cfg=resolve_trpo_regime(TRPOConfig(critic_regime=regime))
    assert (cfg.gae_lambda,cfg.critic_lambda,cfg.value_updates,cfg.value_num_splits)==(lam,critic_lam,epochs,splits)
    a=TRPO.__new__(TRPO);a.cfg=cfg;a.logZ=torch.tensor(0.4)
    a.value_model=nn.Linear(1,1);a._ensure_value_model=lambda model:None
    a._forward_policy_and_graph_out=lambda *args:(None,torch.tensor([[.2],[.3],[.1]]))
    batch=SimpleNamespace(traj_lens=torch.tensor([2,1]),x=torch.zeros(3,1),cond_info=torch.zeros(2,1),
        old_log_p_F=torch.tensor([-.5,-.6,-.7]),log_p_B=torch.tensor([-.2,0.,0.]),
        log_rewards=torch.tensor([1.,2.]),ppo_step_mask=torch.ones(3,dtype=torch.bool))
    targets=a._compute_fixed_targets(None,batch)
    costs=torch.tensor([-.3,-1.2,-2.3]);deltas=costs+torch.tensor([.3,0.,0.])-torch.tensor([.2,.3,.1])
    raw=torch.tensor([deltas[0]+lam*deltas[1],deltas[1],deltas[2]])
    torch.testing.assert_close(targets['advantages'],raw-raw.mean())
    values=torch.tensor([.2,.3,.1]);expected=values+torch.tensor([deltas[0]+critic_lam*deltas[1],deltas[1],deltas[2]])
    torch.testing.assert_close(targets['value_targets'],expected)
    torch.testing.assert_close(targets['cost_mean'],torch.tensor((-0.3-1.2-2.3)/2))
    if regime=='gfn_pg':torch.testing.assert_close(expected,torch.tensor([-1.5,-1.2,-2.3]))


def test_algorithm_checkpoint_roundtrip_and_parameter_exclusion():
    def algorithm():
        a=TRPO.__new__(TRPO);a.cfg=TRPOConfig();a.global_cfg=Config()
        a.value_model=None;a.value_opt=None;a.logZ=None;a.logZ_opt=None
        return a
    model=nn.Module();model.trunk=nn.Linear(2,2);model.emb2graph_out=nn.Linear(2,1)
    first=algorithm();first._ensure_value_model(model)
    ids={id(p) for p in first._policy_parameters(model)}
    assert ids=={id(p) for p in model.trunk.parameters()}
    assert not ids.intersection(id(p) for p in first.value_model.parameters())
    with torch.no_grad(): first.logZ.fill_(3.); next(first.value_model.parameters()).add_(1.)
    state=first.state_dict();second=algorithm();second.load_state_dict(state,model)
    for p,q in zip(first.value_model.parameters(),second.value_model.parameters()):torch.testing.assert_close(p,q)
    torch.testing.assert_close(first.logZ,second.logZ)


@pytest.mark.parametrize('accepted',[False,True])
def test_trainer_updates_ema_only_after_accepted_trpo_step(accepted):
    from gflownet.trainer import GFNTrainer, StandardOnlineTrainer
    model=nn.Linear(1,1,bias=False);ema=nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.fill_(2.);ema.weight.fill_(1.)
    cfg=Config();cfg.algo.valid_use_ema=True;cfg.algo.valid_ema_tau=.95
    def forbidden(*args,**kwargs):raise AssertionError('TRPO must not call Adam policy step')
    trainer=SimpleNamespace(model=model,validation_model=ema,cfg=cfg,_validate_parameters=True,
        step=forbidden,algo=SimpleNamespace(manual_policy_update=True,
        update_batch=lambda *args,**kwargs:{'trpo_accepted':float(accepted)},step=lambda:None))
    trainer._update_validation_model=lambda:StandardOnlineTrainer._update_validation_model(trainer)
    GFNTrainer.train_batch(trainer,SimpleNamespace(),0,0,0)
    torch.testing.assert_close(ema.weight,torch.tensor([[1.05 if accepted else 1.]]))
