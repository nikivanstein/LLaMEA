import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for ind in population:
                performance_ratio = 1.0 - func(ind) / func(best_individual)
                mutation_prob = np.clip(0.2 + 0.6 * performance_ratio, 0.2, 0.8)
                if np.random.rand() < mutation_prob:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.randn(self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]