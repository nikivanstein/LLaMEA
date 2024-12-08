import numpy as np

class FlockingAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def get_fitness(population):
            return np.array([func(solution) for solution in population])

        def update_population(population, fitness):
            new_population = np.copy(population)
            for i in range(self.budget):
                neighbors_idx = np.random.choice(np.delete(np.arange(self.budget), i), 3, replace=False)
                average_neighbor = np.mean(population[neighbors_idx], axis=0)
                new_solution = population[i] + np.random.uniform(-1, 1) * (average_neighbor - population[i])
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_fitness = func(new_solution)
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
            return population

        population = initialize_population()
        fitness = get_fitness(population)

        for _ in range(self.budget - self.budget // 10):
            population = update_population(population, fitness)
            fitness = get_fitness(population)

        best_idx = np.argmin(fitness)
        return population[best_idx]