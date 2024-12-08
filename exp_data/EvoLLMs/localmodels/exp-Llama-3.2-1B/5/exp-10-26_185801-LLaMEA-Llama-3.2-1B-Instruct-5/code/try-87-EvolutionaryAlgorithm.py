import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class HEBBO(EvolutionaryAlgorithm):
    def __init__(self, budget, dim, mutation_rate):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        new_individual = np.copy(self.search_space)
        for _ in range(self.budget // 2):
            if np.random.rand() < self.mutation_rate:
                index = np.random.randint(0, self.dim)
                new_individual[index] += np.random.uniform(-1, 1)
        func_value = self.__call__(func)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

class GeneticAlgorithm(EvolutionaryAlgorithm):
    def __init__(self, budget, dim, mutation_rate):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        population_size = 100
        population = [np.copy(self.search_space) for _ in range(population_size)]
        for _ in range(100):
            for individual in population:
                func_value = self.__call__(func, individual)
                if np.isnan(func_value) or np.isinf(func_value):
                    raise ValueError("Invalid function value")
                if func_value < 0 or func_value > 1:
                    raise ValueError("Function value must be between 0 and 1")
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
                population[population.index(individual)] = np.copy(individual)
        return np.max(population)

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 