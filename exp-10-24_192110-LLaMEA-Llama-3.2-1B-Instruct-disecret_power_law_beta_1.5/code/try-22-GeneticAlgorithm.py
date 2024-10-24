import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [[np.random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        def evaluate(func, x):
            return func(x)

        def fitness_func(x):
            return evaluate(func, x)

        def selection_func(individual, fitness):
            return fitness / fitness.max()

        def crossover_func(ind1, ind2):
            x1, y1 = ind1
            x2, y2 = ind2
            cx1_x = self.crossover(x1, x2)
            cx1_y = self.crossover(y1, y2)
            return [cx1_x, cx1_y]

        def mutation_func(individual):
            x, y = individual
            if np.random.rand() < 0.1:
                x += np.random.uniform(-1.0, 1.0)
            if np.random.rand() < 0.1:
                y += np.random.uniform(-1.0, 1.0)
            return [x, y]

        while len(self.population) > 0:
            # Selection
            self.population = sorted(self.population, key=lambda x: fitness_func(x), reverse=True)[:self.population_size // 2]

            # Crossover
            children = []
            for i in range(self.population_size // 2):
                parent1, parent2 = self.population[i * 2], self.population[i * 2 + 1]
                child = crossover_func(parent1, parent2)
                children.extend(mutation_func(child))

            # Mutation
            for i in range(self.population_size):
                if np.random.rand() < 0.05:
                    children[i] = mutation_func(children[i])

            # Replace with new generation
            self.population = children

        return self.population[0]

    def fitness_func(self, func, x):
        return func(x)