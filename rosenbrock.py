def rosenbrock(x, alpha):
    # Rosenbrock function
    # Used to test optimization algorithms
    '''
    x: array of length 2
    alpha: float
    '''
    return (1 - x[0])**2 + alpha * (x[1] - x[0]**2)**2
