import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        mutation_probs = np.full(self.budget, 0.3)
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for idx, ind in enumerate(population):
                if np.random.rand() < mutation_probs[idx]:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                    mutation_probs[idx] *= 0.9  # Dynamic mutation adjustment
                else:
                    new_ind = ind + np.random.randn(self.dim)
                    mutation_probs[idx] = max(mutation_probs[idx], 0.3)  # Reset to default
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]