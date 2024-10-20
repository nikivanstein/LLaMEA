import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        # Initialize the population
        population = np.random.rand(self.budget, self.dim)

        # Evaluate the fitness of the population
        fitness = np.apply_along_axis(func, 1, population)

        # Select the best solution
        best_index = np.argmin(fitness)
        self.best_solution = population[best_index]
        self.best_fitness = fitness[best_index]

        # Update the population
        for i in range(self.budget):
            # Select two random solutions
            r1 = np.random.randint(0, self.budget)
            r2 = np.random.randint(0, self.budget)

            # Crossover
            offspring = np.where(np.random.rand(self.dim) < 0.5, population[r1], population[r2])

            # Mutation
            offspring = np.where(np.random.rand(self.dim) < 0.1, np.random.rand(self.dim), offspring)

            # Evaluate the fitness of the offspring
            offspring_fitness = np.apply_along_axis(func, 1, offspring)

            # Select the best offspring
            best_offspring_index = np.argmin(offspring_fitness)
            offspring = offspring[best_offspring_index]

            # Update the population
            population[i] = offspring

        return self.best_solution