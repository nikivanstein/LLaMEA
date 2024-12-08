import numpy as np

class HybridCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pa = 0.25  # Probability of abandoning a nest
        self.alpha = 1.5  # Levy flight parameter
        self.nests = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        
    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.alpha) * np.math.sin(np.pi * self.alpha / 2) / (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / self.alpha)
        return 0.01 * step
        
    def __call__(self, func):
        best_nest = self.nests[np.argmin([func(nest) for nest in self.nests])]
        
        for _ in range(self.budget):
            new_nests = []
            for nest in self.nests:
                if np.random.rand() < self.pa:
                    step_size = self.levy_flight()
                    new_nest = nest + step_size * (nest - best_nest)
                    new_nests.append(new_nest)
                else:
                    new_nests.append(nest)
                    
            new_nests = np.clip(new_nests, self.lower_bound, self.upper_bound)
            self.nests = sorted(new_nests, key=lambda x: func(x))[:self.budget]
            best_nest = self.nests[np.argmin([func(nest) for nest in self.nests])]
        
        return best_nest