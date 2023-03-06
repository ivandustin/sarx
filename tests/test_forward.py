from jax.numpy import array, array_equal
from pytest import fixture
from sarx import spike, forward


@fixture
def x():
    return array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 2.0],
        [0.0, 0.0]
    ])


@fixture
def S():
    return [
        array([
            [1.0, 0.1],
            [0.2, 1.0]
        ]),
        array([
            [0.5]
        ])
    ]


@fixture
def expected_a():
    return array([
        [
            [1.2],
            [1.0],
            [0.2],
            [2.4],
            [0.0]
        ],
        [
            [1.7],
            [0.6],
            [1.0],
            [3.2],
            [0.0]
        ]
    ])


@fixture
def expected_b():
    return array([
        [
            [1.2],
            [1.0],
            [0.0],
            [2.0],
            [0.0]
        ],
        [
            [1.7],
            [0.0],
            [1.0],
            [2.0],
            [0.0]
        ]
    ])


def test_1(S, x, expected_a, expected_b):
    actual = forward(spike)(S, x)
    actual_a = array(actual[0])
    actual_b = array(actual[1])
    assert array_equal(actual_a, expected_a)
    assert array_equal(actual_b, expected_b)


def test_2(S, x, expected_a, expected_b):
    actual = forward(S, x)
    actual_a = array(actual[0])
    actual_b = array(actual[1])
    assert array_equal(actual_a, expected_a)
    assert array_equal(actual_b, expected_b)
