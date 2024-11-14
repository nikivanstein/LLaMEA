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
            for i, ind in enumerate(population):
                if i == best_idx:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.normal(0, mutation_probs[i], self.dim)
                new_population.append(new_ind)
                if func(new_ind) < func(ind):
                    mutation_probs[i] *= 1.05  # Increase mutation probability for better individuals
                else:
                    mutation_probs[i] *= 0.95  # Decrease mutation probability for worse individuals
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]