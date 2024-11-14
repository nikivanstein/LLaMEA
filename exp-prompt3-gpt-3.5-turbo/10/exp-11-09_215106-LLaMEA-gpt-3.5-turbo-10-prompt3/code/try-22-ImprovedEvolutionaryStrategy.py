import numpy as np

class ImprovedEvolutionaryStrategy(EvolutionaryStrategy):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mu_min = 5  # minimum population size
        self.mu_max = 20  # maximum population size
        
    def __call__(self, func):
        mu_min = self.mu_min
        mu_max = self.mu_max
        
        # Dynamic population adaptation
        for _ in range(self.budget // self.lambda_):
            self.mu = mu_min + int((mu_max - mu_min) * _ / (self.budget // self.lambda_))
            super().__call__(func)
        
        return x[0]