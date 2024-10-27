import numpy as np

class HybridBOBOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.01, crossover_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def __call__(self, func, initial_individual, logger):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.evaluate_fitness(initial_individual, func, logger)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid function value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if np.random.rand() < self.mutation_rate:
                new_individual = self.mutation(new_individual, func, self.search_space, logger)
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = self.select_parents(initial_individual, new_individual)
                child = self.crossover(parent1, parent2, func, logger)
                new_individual = self.evaluate_fitness(child, func, logger)
            self.func_evaluations = 0
            self.func_evaluations += 1
        return new_individual

    def mutation(self, individual, func, search_space, logger):
        new_individual = individual + np.random.uniform(-1, 1, self.dim)
        while np.isnan(new_individual) or np.isinf(new_individual):
            new_individual = individual + np.random.uniform(-1, 1, self.dim)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid function value")
        func_value = func(new_individual)
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return new_individual

    def crossover(self, parent1, parent2, func, logger):
        if np.random.rand() < self.crossover_rate:
            return parent1
        else:
            return parent2

    def select_parents(self, individual, new_individual):
        # Simple selection strategy: choose the better parent based on the fitness values
        # This can be improved by using a more sophisticated selection strategy
        if np.isnan(new_individual) or np.isinf(new_individual):
            return individual, new_individual
        if np.isnan(individual) or np.isinf(individual):
            return new_individual, individual
        return individual, new_individual

    def evaluateBBOB(self, func, logger):
        # Evaluate the function for a given individual
        func_value = func(self.search_space)
        logger.update(func_value)
        return func_value