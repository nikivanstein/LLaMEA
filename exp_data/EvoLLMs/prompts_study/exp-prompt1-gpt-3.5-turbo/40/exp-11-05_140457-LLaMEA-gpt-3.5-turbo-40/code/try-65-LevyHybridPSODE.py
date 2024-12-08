import numpy as np

class LevyHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.local_search_radius = 0.1
        self.levy_scale = 0.1
        
    def levy_flight(self):
        beta = 1.5
        sigma = np.power(np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / np.math.gamma((1 + beta) / 2) / beta, 1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1 / beta)
        return step * self.levy_scale
    
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = self.levy_flight()
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best