from jax import grad


def feed(loss, update):
    def function(network, x, y):
        gradient = grad(loss)(network, x, y)
        return update(network, gradient)
    return function
