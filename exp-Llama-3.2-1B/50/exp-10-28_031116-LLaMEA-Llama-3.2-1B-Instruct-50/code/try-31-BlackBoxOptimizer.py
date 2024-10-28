import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))

    def __call__(self, func):
        def _fitness_func(x):
            return func(x)

        def _bounds_func(x):
            return np.clip(x, -5.0, 5.0)

        def _bounds(x):
            return _bounds_func(x)

        def _select_parent(parents):
            return parents[np.random.randint(0, len(parents))]

        def _mutate(parent, mutation_rate):
            x = self._select_parent(parent)
            if np.random.rand() < mutation_rate:
                x += np.random.uniform(-1.0, 1.0)
            return x

        def _crossover(parent1, parent2):
            return np.concatenate((parent1[:len(parent1)//2], parent2[len(parent2)//2:]))

        def _mutate_and_crossover(parent, child, mutation_rate):
            child = self._mutate(parent, mutation_rate)
            while np.random.rand() < mutation_rate:
                child = self._mutate_and_crossover(child, np.random.uniform(-1.0, 1.0), mutation_rate)
            return child

        def _select_neighbor(parent, neighbor):
            return np.random.choice([parent, neighbor], p=[0.5, 0.5])

        def _mutate_and_select_neighbor(parent, neighbor, mutation_rate):
            child = self._mutate_and_crossover(parent, neighbor, mutation_rate)
            while np.random.rand() < mutation_rate:
                child = self._mutate_and_select_neighbor(child, np.random.uniform(-1.0, 1.0), mutation_rate)
            return child

        def _evaluate(population):
            fitness = np.zeros(len(population))
            for i, x in enumerate(population):
                fitness[i] = _fitness_func(x)
            return fitness

        fitness = _evaluate(self.population)
        if np.any(fitness < 0):
            self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))

        best_x = np.min(self.population, axis=0)
        best_fitness = np.min(fitness)
        return best_x, best_fitness

    def run(self, func, bounds, mutation_rate=0.1, selection_rate=0.5, crossover_rate=0.1, neighborhood_size=5):
        while True:
            population = self.population
            fitness = np.zeros((len(population), self.dim))
            for i, x in enumerate(population):
                fitness[i] = _fitness_func(x)
            if np.any(fitness < 0):
                population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

            best_x, best_fitness = np.min(population, axis=0), np.min(fitness)
            if best_fitness >= self.budget:
                break

            parent = np.random.choice(len(population), p=[1 - selection_rate, selection_rate])
            child = self._mutate_and_crossover(population[parent], population[parent], crossover_rate)
            neighbor = self._select_neighbor(population[parent], population[parent])
            child = self._mutate_and_select_neighbor(child, neighbor, neighborhood_size)

            population = np.concatenate((population[parent], child, population[neighbor]), axis=0)

        return best_x, best_fitness