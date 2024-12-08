import numpy as np

class AdaptiveStepSizeES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.step_size = 0.1  # Initial step size

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            new_population = []
            for ind in population:
                step_size = self.step_size * np.exp(0.01 * (fitness[best_idx] - func(ind)))  # Adaptive step size adjustment
                mutated = ind + step_size * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual