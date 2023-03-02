def loss(infer, loss):
    def function(network, x, y):
        yhat = infer(network, x)
        if callable(y):
            y = y(yhat)
        return loss(y, yhat)
    return function
