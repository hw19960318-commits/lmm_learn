import pytest
import torch
from common import make_batch


def test_saved_and_reloaded_outputs_match(impl,tmp_path):
    model=impl('model').TinyRegressor(); x,y=make_batch()
    opt=torch.optim.SGD(model.parameters(),lr=0.05)
    for _ in range(20):
        impl('training').train_step(model,opt,x,y)
    model.eval()
    with torch.no_grad():
        before=model(x).clone()
    path=tmp_path/'nested'/'model.pt'
    impl('checkpoint').save_model(model,path)
    restored=impl('checkpoint').load_model(path)
    with torch.no_grad():
        after=restored(x)
    torch.testing.assert_close(before,after,rtol=1e-6,atol=1e-7)
    assert restored.training is False
    assert next(restored.parameters()).device.type=='cpu'


def test_checkpoint_contains_state_dict(impl,tmp_path):
    path=tmp_path/'model.pt'
    impl('checkpoint').save_model(impl('model').TinyRegressor(),path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert set(state)=={'linear.weight','linear.bias'}


def test_loading_missing_file_is_clear(impl,tmp_path):
    with pytest.raises(FileNotFoundError):
        impl('checkpoint').load_model(tmp_path/'missing.pt')


def test_reloaded_model_is_independent(impl,tmp_path):
    original=impl('model').TinyRegressor(); path=tmp_path/'model.pt'
    impl('checkpoint').save_model(original,path)
    restored=impl('checkpoint').load_model(path)
    saved_weight=restored.linear.weight.detach().clone()
    with torch.no_grad():
        original.linear.weight.add_(100)
    torch.testing.assert_close(restored.linear.weight,saved_weight)
