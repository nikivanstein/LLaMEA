# import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim, param=0.5):
        self.budget = budget
        self.dim = dim
        self.param = param

    def __call__(self, func):
        def initialize_population(pop_size):
            return np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

        def update_population(population, fitnesses):
            idx = np.argmin(fitnesses)
            return population[idx]

        pop_size = 10
        population = initialize_population(pop_size)
        fitness = np.apply_along_axis(func, 1, population)

        for _ in range(self.budget - pop_size):
            new_solution = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.param:
                    new_solution[i] = np.random.choice(population)[i]
                else:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
            new_fitness = func(new_solution)

            if new_fitness < np.max(fitness):
                population = np.vstack((population, new_solution))
                fitness = np.append(fitness, new_fitness)
                if len(population) > pop_size:
                    idx = np.argmax(fitness)
                    population = np.delete(population, idx, axis=0)
                    fitness = np.delete(fitness, idx)
        
        best_solution = update_population(population, fitness)

        return best_solution