from jax import grad
from sarx import spike


def test_spike_negative():
    assert spike(-1.0) == 0.0


def test_spike_zero():
    assert spike(0.0) == 0.0


def test_spike_below_one():
    assert spike(0.5) == 0.0


def test_spike_one():
    assert spike(1.0) == 1.0


def test_spike_above_one():
    assert spike(1.5) == 1.5


def test_spike_two():
    assert spike(2.0) == 2.0


def test_spike_above_two():
    assert spike(2.5) == 2.0


def test_grad_negative():
    assert grad(spike)(-1.0) == 1.0


def test_grad_zero():
    assert grad(spike)(0.0) == 1.0


def test_grad_below_one():
    assert grad(spike)(0.5) == 1.0


def test_grad_one():
    assert grad(spike)(1.0) == 1.0


def test_grad_above_one():
    assert grad(spike)(1.5) == 1.0


def test_grad_two():
    assert grad(spike)(2.0) == 1.0


def test_grad_above_two():
    assert grad(spike)(2.5) == 1.0
