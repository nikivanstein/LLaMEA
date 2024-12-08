import numpy as np

class GASimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iters = budget // self.population_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population(dim, population_size, lb, ub):
            return np.random.uniform(lb, ub, (population_size, dim))

        def clip_position(p, lb, ub):
            return np.clip(p, lb, ub)

        def select_parents(population, fitness_values):
            idx = np.argsort(fitness_values)
            return population[idx[:2]]

        def mutate(parent, lb, ub):
            return parent + np.random.normal(0, 1, parent.shape)

        population = initialize_population(self.dim, self.population_size, self.lb, self.ub)
        for _ in range(self.max_iters):
            fitness_values = np.array([func(individual) for individual in population])
            parents = select_parents(population, fitness_values)
            child = mutate(np.mean(parents, axis=0), self.lb, self.ub)
            population = np.vstack((population, clip_position(child, self.lb, self.ub)))
        best_individual = population[np.argmin([func(individual) for individual in population])]
        return best_individual