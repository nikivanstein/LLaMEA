import numpy as np

class EnhancedAISOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10
        self.max_iterations = budget // self.initial_population_size

    def __call__(self, func):
        def initialize_population(population_size):
            return np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))

        def mutate_population(population, diversity, best_survivor):
            mutation_factor = 1 + 0.1 * np.exp(-0.1 * np.linalg.norm(population - best_survivor, axis=1))
            return population + np.random.normal(0, diversity * mutation_factor, size=population.shape)

        def elitism_selection(population, func):
            scores = np.apply_along_axis(func, 1, population)
            best_idx = np.argmin(scores)
            return population[best_idx]

        population_size = self.initial_population_size
        population = initialize_population(population_size)
        diversity = 1.0
        for _ in range(self.max_iterations):
            best_survivor = elitism_selection(population, func)
            mutated_population = mutate_population(population, diversity, best_survivor)
            survivors = np.vstack((population, best_survivor))
            survivors = survivors[np.argsort(np.apply_along_axis(func, 1, survivors))][:population_size]
            best_survivor = elitism_selection(survivors, func)
            population = np.vstack((population, best_survivor))
            population_size = max(1, min(2 * population_size, self.budget // len(population)))
            population = population[:population_size]
            diversity = len(np.unique(population)) / len(population)
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution