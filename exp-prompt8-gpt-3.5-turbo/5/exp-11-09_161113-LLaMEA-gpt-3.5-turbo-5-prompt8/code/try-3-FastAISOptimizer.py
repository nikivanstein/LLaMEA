import numpy as np

class FastAISOptimizer(AISOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def mutate_population(population, iteration):
            mutation_scale = 1.0 / (1.0 + iteration / self.max_iterations)  # Dynamic mutation scaling
            return population + mutation_scale * np.random.normal(0, 1, size=population.shape)

        def select_survivors(current_population, mutated_population, func):
            scores_current = np.apply_along_axis(func, 1, current_population)
            scores_mutated = np.apply_along_axis(func, 1, mutated_population)
            return current_population[scores_current < scores_mutated]

        population = initialize_population()
        for iteration in range(self.max_iterations):
            mutated_population = mutate_population(population, iteration)
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            population = np.vstack((population, survivors))
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution