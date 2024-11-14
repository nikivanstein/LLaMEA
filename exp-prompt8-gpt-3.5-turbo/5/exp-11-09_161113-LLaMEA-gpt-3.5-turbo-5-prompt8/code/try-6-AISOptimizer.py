import numpy as np

class AISOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.max_iterations = budget // self.population_size
        self.min_step_size = 0.01
        self.max_step_size = 1.0
        self.step_size = self.max_step_size

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def mutate_population(population):
            return population + np.random.normal(0, self.step_size, size=population.shape)

        def select_survivors(current_population, mutated_population, func):
            scores_current = np.apply_along_axis(func, 1, current_population)
            scores_mutated = np.apply_along_axis(func, 1, mutated_population)
            return current_population[scores_current < scores_mutated]

        population = initialize_population()
        for _ in range(self.max_iterations):
            mutated_population = mutate_population(population)
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            population = np.vstack((population, survivors))
            success_rate = len(survivors) / len(population)
            self.step_size = min(self.max_step_size, max(self.min_step_size, self.step_size * (1 + 0.2 * (success_rate - 0.5))))
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution