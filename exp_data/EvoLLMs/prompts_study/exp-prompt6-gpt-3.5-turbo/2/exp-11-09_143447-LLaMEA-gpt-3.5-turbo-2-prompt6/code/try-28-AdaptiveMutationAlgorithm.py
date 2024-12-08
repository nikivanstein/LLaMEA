import numpy as np

class AdaptiveMutationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1  # Initial mutation rate
        self.history = []  # List to store historical information

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            new_population = []
            for ind in population:
                mutation_rate = self.mutation_rate + 0.01 * (fitness[best_idx] - func(ind))  # Dynamic mutation adjustment
                self.history.append(mutation_rate)  # Store mutation rate in history
                if len(self.history) > 1:
                    mutation_rate += 0.1 * (self.history[-1] - self.history[-2])  # Adapt mutation rate based on historical information
                mutated = ind + mutation_rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual