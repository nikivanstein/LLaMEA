import numpy as np
import concurrent.futures

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def evaluate_fitness(population):
            return np.apply_along_axis(func, 1, population)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            fitness = evaluate_fitness(population)

        best_solution = population[np.argmin(fitness)]
        return best_solution