from sarx import gd


def test_gd():
    theta = 1.0
    gradient = 2.0
    alpha = 3.0
    assert gd(theta, gradient, alpha) == theta - alpha * gradient
