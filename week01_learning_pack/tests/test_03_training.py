import pytest
import torch
from common import make_batch


def test_model_shape_and_parameter_count(impl):
    model=impl('model').TinyRegressor()
    x,_=make_batch()
    assert model(x).shape == (32,1)
    assert sum(p.numel() for p in model.parameters()) == 3
    assert model.linear.weight.shape == (1,2)


def test_one_step_really_changes_weights(impl):
    model=impl('model').TinyRegressor(); x,y=make_batch()
    old=[p.detach().clone() for p in model.parameters()]
    opt=torch.optim.SGD(model.parameters(),lr=0.05)
    log=impl('training').train_step(model,opt,x,y)
    assert log['grad_norm']>0 and log['parameter_delta']>0
    assert any(not torch.equal(p,oldp) for p,oldp in zip(model.parameters(),old))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_single_batch_loss_falls(impl):
    model=impl('model').TinyRegressor(); x,y=make_batch(); train=impl('training')
    opt=torch.optim.SGD(model.parameters(),lr=0.05)
    initial=train.evaluate(model,x,y)
    for _ in range(200):
        train.train_step(model,opt,x,y)
    final=train.evaluate(model,x,y)
    assert final < initial * 0.01
    assert final < 1e-3
    torch.testing.assert_close(model.linear.weight.detach(),torch.tensor([[2.,-3.]]),atol=0.03,rtol=0)


def test_zero_learning_rate_does_not_change_weights(impl):
    model=impl('model').TinyRegressor(); x,y=make_batch()
    old=[p.detach().clone() for p in model.parameters()]
    opt=torch.optim.SGD(model.parameters(),lr=0.0)
    impl('training').train_step(model,opt,x,y)
    assert all(torch.equal(p,o) for p,o in zip(model.parameters(),old))


def test_evaluation_does_not_change_parameters_or_gradients(impl):
    model=impl('model').TinyRegressor(); x,y=make_batch()
    old=[p.detach().clone() for p in model.parameters()]
    result=impl('training').evaluate(model,x,y)
    assert isinstance(result,float)
    assert all(torch.equal(p,o) for p,o in zip(model.parameters(),old))
    assert all(p.grad is None for p in model.parameters())
    assert model.training is False


def test_training_loss_rejects_wrong_target_shape(impl):
    model=impl('model').TinyRegressor(); x,y=make_batch()
    opt=torch.optim.SGD(model.parameters(),lr=0.05)
    with pytest.raises(ValueError):
        impl('training').train_step(model,opt,x,y.squeeze(1))


def test_eval_mode_is_not_no_grad():
    # 提供的概念测试：不需要自己实现；读懂即可。
    model=torch.nn.Linear(2,1).eval()
    x=torch.ones(1,2)
    assert model(x).requires_grad
    with torch.no_grad():
        assert not model(x).requires_grad
