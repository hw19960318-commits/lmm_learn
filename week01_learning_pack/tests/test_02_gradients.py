import pytest
import torch


def test_scalar_example_matches_hand_derivation(impl):
    r = impl('autograd_lab').single_example_gradients()
    assert r['prediction'] == pytest.approx(2.0)
    assert r['loss'] == pytest.approx(9.0)
    assert r['grad_w'] == pytest.approx(-12.0)
    assert r['grad_b'] == pytest.approx(-6.0)


def test_backward_does_not_update_parameters(impl):
    r = impl('autograd_lab').single_example_gradients()
    assert r['w_after_backward'] == pytest.approx(1.0)
    assert r['b_after_backward'] == pytest.approx(0.0)


def test_gradients_for_different_parameters_not_hardcoded(impl):
    w,b = 0.25,-0.75
    r = impl('autograd_lab').single_example_gradients(w,b)
    residual = 2*w+b-5
    assert r['grad_w'] == pytest.approx(4*residual)
    assert r['grad_b'] == pytest.approx(2*residual)


def test_accumulation_and_reset(impl):
    a,b,c = impl('autograd_lab').accumulation_demo()
    assert (a,b,c) == pytest.approx((2.,4.,2.))


def test_finite_difference_matches_analytic_gradient():
    # 提供的扩展测试：数值差分仅用于小标量检查，不替代正常反向传播。
    f = lambda w: (2*w-5)**2
    eps = 1e-5
    grad = (f(1+eps)-f(1-eps))/(2*eps)
    assert grad == pytest.approx(-12, rel=1e-6)
