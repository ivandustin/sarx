from jax import grad
from sarx import mae


def test_mae_negative():
    assert mae(-2.0) == 2.0

def test_mae_positive():
    assert mae(2.0) == 2.0

def test_mae_zero():
    assert mae(0.0) == 0.0

def test_grad_negative():
    assert grad(mae)(-2.0) == -1.0

def test_grad_positive():
    assert grad(mae)(2.0) == 1.0

def test_grad_zero():
    assert grad(mae)(0.0) == 0.0
