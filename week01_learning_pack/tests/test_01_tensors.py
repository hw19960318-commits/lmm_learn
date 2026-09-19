import pytest
import torch


def test_creation_shape_values_and_dtype(impl):
    x = impl('tensor_ops').create_demo_tensor()
    assert x.shape == (3,4) and x.dtype == torch.float32
    torch.testing.assert_close(x, torch.arange(12, dtype=torch.float32).reshape(3,4))


def test_last_column_preserves_dimension(impl):
    x = torch.arange(12).reshape(3,4)
    got = impl('tensor_ops').last_column(x)
    assert got.shape == (3,1)
    assert got.tolist() == [[3],[7],[11]]


def test_last_column_rejects_invalid_input(impl):
    with pytest.raises(ValueError):
        impl('tensor_ops').last_column(torch.ones(3))


def test_affine_known_values(impl):
    x = torch.tensor([[1.,2.],[3.,4.]])
    w = torch.tensor([[2.],[-3.]])
    got = impl('tensor_ops').affine(x,w,torch.tensor([0.5]))
    torch.testing.assert_close(got, torch.tensor([[-3.5],[-5.5]]))


def test_affine_multiple_outputs(impl):
    x = torch.ones(5,3); w = torch.ones(3,2); b = torch.tensor([1.,2.])
    torch.testing.assert_close(impl('tensor_ops').affine(x,w,b),torch.tensor([[4.,5.]]).repeat(5,1))


def test_mse_is_scalar_and_correct(impl):
    got = impl('tensor_ops').mse(torch.tensor([[1.],[3.]]),torch.tensor([[2.],[1.]]))
    assert got.ndim == 0 and got.item() == pytest.approx(2.5)


def test_mse_rejects_silent_broadcast(impl):
    with pytest.raises(ValueError):
        impl('tensor_ops').mse(torch.ones(4,1),torch.ones(4))


def test_mse_rejects_empty_batch(impl):
    with pytest.raises(ValueError):
        impl('tensor_ops').mse(torch.empty(0,1),torch.empty(0,1))
