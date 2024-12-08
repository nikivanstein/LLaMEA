import numpy as np

class EnhancedAdaptiveMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_probs = np.ones(dim) * 0.5
        self.step_size = 0.1  # Initial mutation step size

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]

            new_population = []
            for individual in self.population:
                mutant = individual + np.random.normal(0, self.mutation_probs * self.step_size)
                new_population.append(mutant)

            self.population = np.array(new_population)
            self.step_size *= 0.95  # Adjust step size dynamically based on population performance

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual