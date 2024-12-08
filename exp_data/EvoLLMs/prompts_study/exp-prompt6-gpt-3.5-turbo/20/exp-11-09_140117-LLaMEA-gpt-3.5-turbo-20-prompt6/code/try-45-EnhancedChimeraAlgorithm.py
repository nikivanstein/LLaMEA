import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        mutation_probs = np.full(self.budget, 0.5)  # Initialize mutation probabilities
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for ind, mut_prob in zip(population, mutation_probs):
                if np.random.rand() < mut_prob:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.randn(self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
            mutation_improvement = np.abs(func(best_individual) - func(population[best_idx])) / func(best_individual)
            mutation_probs = np.clip(mutation_probs * (1 + mutation_improvement), 0, 1)  # Update mutation probabilities
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]