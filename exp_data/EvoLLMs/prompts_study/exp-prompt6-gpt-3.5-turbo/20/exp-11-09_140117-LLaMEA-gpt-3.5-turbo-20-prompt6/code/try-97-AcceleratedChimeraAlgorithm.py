import numpy as np

class AcceleratedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for idx, ind in enumerate(population):
                mutation_prob = 0.3 + 0.7 * (1 - idx / self.budget)  # Dynamic mutation probability
                if np.random.rand() < mutation_prob:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.randn(self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]