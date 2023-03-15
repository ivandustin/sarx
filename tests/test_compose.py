from sarx import compose


def test():
    assert compose(increment)(0) == 1
    assert compose(increment, increment)(0) == 2
    assert compose(increment, increment, increment)(0) == 3


def test_multiple_args():
    assert compose(add)(1, 2) == 3
    assert compose(increment, add)(1, 2) == 4


def increment(x):
    return x + 1


def add(a, b):
    return a + b
