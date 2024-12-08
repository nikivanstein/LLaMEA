import numpy as np

class DynamicChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.full(self.budget, 0.5)  # Initialize mutation rates to 0.5
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for idx, ind in enumerate(population):
                mutation_prob = np.random.rand()
                if mutation_prob < self.mutation_rates[idx]:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.randn(self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
            # Update mutation rates based on individual performance
            fitness_values = [func(ind) for ind in population]
            self.mutation_rates = 0.5 + 0.5 * (1.0 - np.array(fitness_values) / np.max(fitness_values))
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]