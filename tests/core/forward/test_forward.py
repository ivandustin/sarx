from jax.numpy import array, array_equal
from sarx.core.forward import forward
from sarx import spike


def test():
    x = array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 2.0],
        [0.0, 0.0]
    ])
    S = [
        array([
            [1.0, 0.1],
            [0.2, 1.0]
        ]),
        array([
            [0.5]
        ])
    ]
    expected_a = array([
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
    expected_b = array([
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
    actual = forward(spike)(S, x)
    actual_a = array(actual[0])
    actual_b = array(actual[1])
    assert array_equal(actual_a, expected_a)
    assert array_equal(actual_b, expected_b)
