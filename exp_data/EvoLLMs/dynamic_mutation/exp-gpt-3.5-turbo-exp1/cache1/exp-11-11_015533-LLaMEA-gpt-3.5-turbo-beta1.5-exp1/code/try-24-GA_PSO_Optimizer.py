class GA_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Remaining code stays the same as the initial solution