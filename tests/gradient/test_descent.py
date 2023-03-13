from sarx.gradient import descent


def test_gd():
    theta = 1.0
    gradient = 2.0
    alpha = 3.0
    assert descent(theta, gradient, alpha) == theta - alpha * gradient
