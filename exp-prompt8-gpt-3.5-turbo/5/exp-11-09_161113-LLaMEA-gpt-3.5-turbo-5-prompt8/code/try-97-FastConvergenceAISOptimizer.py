import numpy as np

class FastConvergenceAISOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10
        self.max_iterations = budget // self.initial_population_size

    def __call__(self, func):
        def initialize_population(population_size):
            return np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))

        def mutate_population(population, diversity):
            return population + np.random.normal(0, 1 + diversity, size=population.shape)

        def select_survivors(current_population, mutated_population, func):
            scores_current = np.apply_along_axis(func, 1, current_population)
            scores_mutated = np.apply_along_axis(func, 1, mutated_population)
            return current_population[scores_current < scores_mutated]

        def elitism_selection(population, func):
            scores = np.apply_along_axis(func, 1, population)
            best_idx = np.argmin(scores)
            best_solution = population[best_idx]

            # Implementing probabilistic elitism based on modified gamma distribution
            acceptance_prob = np.exp(-func(best_solution) / np.sum(func(population)))
            if np.random.rand() < acceptance_prob:
                return best_solution
            else:
                return population[np.random.randint(len(population))]

        population_size = self.initial_population_size
        population = initialize_population(population_size)
        diversity = 1.0
        for _ in range(self.max_iterations):
            mutated_population = mutate_population(population, diversity)
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            best_survivor = elitism_selection(survivors, func)
            population = np.vstack((population, best_survivor))
            population_size = max(1, min(2 * population_size, self.budget // len(population)))
            population = population[:population_size]
            diversity = len(np.unique(population)) / len(population)
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution