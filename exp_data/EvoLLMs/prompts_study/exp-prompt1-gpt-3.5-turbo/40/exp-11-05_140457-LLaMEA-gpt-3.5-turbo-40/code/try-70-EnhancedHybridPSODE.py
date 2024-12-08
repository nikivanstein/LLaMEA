import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.local_search_radius = 0.1
        
    def __call__(self, func):
        fitness_improvement_trend = []
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
                        fitness_improvement_trend.append(1)
                    else:
                        fitness_improvement_trend.append(0)
            
            else:  # Original HybridPSODE update
                super().__call__(func)
                
                if len(fitness_improvement_trend) >= 3:
                    if sum(fitness_improvement_trend[-3:]) == 0:
                        self.local_search_radius *= 1.1  # Increase radius for exploration
                    elif sum(fitness_improvement_trend[-3:]) == 3:
                        self.local_search_radius *= 0.9  # Decrease radius for exploitation
                    fitness_improvement_trend = []
        return self.global_best