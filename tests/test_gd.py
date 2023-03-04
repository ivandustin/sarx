from sarx import gd


def test_gd():
    assert gd(1.0)(2.0, 3.0) == 2.0 - 1.0 * 3.0
