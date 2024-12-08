import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        mutation_probs = np.full(self.budget, 0.6)  # Initialize mutation probabilities
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for i, ind in enumerate(population):
                if i != best_idx:
                    mutation_prob = np.clip(np.abs(func(ind) - func(best_individual)) / np.abs(func(population[i]) - func(best_individual), 0.3, 0.9)
                    mutation_probs[i] = mutation_prob
                if np.random.rand() < mutation_probs[i]:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.randn(self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]