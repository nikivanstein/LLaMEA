import numpy as np

class DynamicIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def explore_phase(population, best_idx):
        mutation_rates = np.random.uniform(0.05, 0.2, self.dim) * np.abs(best_idx - np.arange(self.dim))
        new_population = population + np.random.normal(0, mutation_rates, population.shape)
        return np.clip(new_population, self.lower_bound, self.upper_bound)