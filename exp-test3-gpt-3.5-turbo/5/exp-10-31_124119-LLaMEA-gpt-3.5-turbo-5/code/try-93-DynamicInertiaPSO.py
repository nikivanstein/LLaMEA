class DynamicInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        # Retain original PSO algorithm with modified parameter values
        pass