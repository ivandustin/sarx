from jax.numpy import array, array_equal
from sarx import spike, infer


def test():
    x = array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
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
    expected = array([
        [
            [1.2],
            [1.0],
            [0.0],
            [0.0]
        ],
        [
            [1.7],
            [0.0],
            [1.0],
            [0.0]
        ]
    ])
    actual = array(infer(spike)(x, S))
    assert array_equal(actual, expected)
