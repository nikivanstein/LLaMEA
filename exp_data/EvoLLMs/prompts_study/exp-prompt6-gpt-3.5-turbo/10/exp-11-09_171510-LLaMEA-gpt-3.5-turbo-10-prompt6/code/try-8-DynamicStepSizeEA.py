import numpy as np

class DynamicStepSizeEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        mutation_step = 0.1
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]

            new_population = []
            for individual in self.population:
                mutant = individual + np.random.normal(0, mutation_step, self.dim)
                new_population.append(mutant)

            self.population = np.array(new_population)

            # Adjust mutation step based on fitness difference
            fitness_diff = np.abs(fitness - np.mean(fitness))
            mutation_step *= np.exp(np.mean(fitness_diff))

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual