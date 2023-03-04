def gd(alpha):
    def function(theta, gradient):
        return theta - alpha * gradient
    return function
