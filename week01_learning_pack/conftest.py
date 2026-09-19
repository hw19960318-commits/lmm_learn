import importlib
import torch
import pytest


def pytest_addoption(parser):
    parser.addoption('--impl', action='store', choices=['exercises', 'reference'], default='exercises')


@pytest.fixture
def impl(request):
    prefix = request.config.getoption('--impl')
    return lambda name: importlib.import_module(f'{prefix}.{name}')


@pytest.fixture(autouse=True)
def deterministic_tiny_run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
