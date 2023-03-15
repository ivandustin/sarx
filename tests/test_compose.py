from sarx import compose


def test():
    assert compose(increment)(0) == 1
    assert compose(increment, increment)(0) == 2
    assert compose(increment, increment, increment)(0) == 3


def increment(x):
    return x + 1
