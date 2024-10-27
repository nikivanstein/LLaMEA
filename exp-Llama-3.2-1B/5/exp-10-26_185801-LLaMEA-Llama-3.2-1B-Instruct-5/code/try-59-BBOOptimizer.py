# BBOB Optimizer: Black Box Optimization using Black Box Optimization
# Description: An evolutionary algorithm that optimizes a black box function using a combination of mutation and crossover operators.

import numpy as np

class BBOOptimizer:
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

class HEBBO(BBOOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Initialize the population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for i in range(self.budget):
            population[i] = func(population[i])

        # Perform crossover and mutation operators
        for i in range(self.budget // 2):
            parent1, parent2 = population[i], population[i + self.budget // 2]
            child = (parent1 + parent2) / 2
            if np.random.rand() < 0.5:
                # Mutation operator: swap two random genes
                idx1, idx2 = np.random.choice(self.dim, 2, replace=False)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            population[i] = child

        # Evaluate the fitness of each individual
        fitnesses = [self.__call__(func) for func in population]

        # Select the fittest individuals
        selected_individuals = np.argsort(fitnesses)[-self.budget:]

        # Refine the selected solution
        selected_individuals = selected_individuals[:self.dim]
        selected_individuals = np.random.choice(selected_individuals, self.budget, replace=False)

        # Replace the selected individuals in the population
        population = np.concatenate((population[:self.budget - self.dim], selected_individuals), axis=0)

        return population

class HEBBO2(BBOOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Initialize the population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for i in range(self.budget):
            population[i] = func(population[i])

        # Perform crossover and mutation operators
        for i in range(self.budget // 2):
            parent1, parent2 = population[i], population[i + self.budget // 2]
            child = (parent1 + parent2) / 2
            if np.random.rand() < 0.5:
                # Mutation operator: swap two random genes
                idx1, idx2 = np.random.choice(self.dim, 2, replace=False)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            population[i] = child

        # Evaluate the fitness of each individual
        fitnesses = [self.__call__(func) for func in population]

        # Select the fittest individuals
        selected_individuals = np.argsort(fitnesses)[-self.budget:]

        # Refine the selected solution
        selected_individuals = selected_individuals[:self.dim]
        selected_individuals = np.random.choice(selected_individuals, self.budget, replace=False)

        # Replace the selected individuals in the population
        population = np.concatenate((population[:self.budget - self.dim], selected_individuals), axis=0)

        return population

# Test the optimizer
def test_hebbbo():
    optimizer = HEBBO(100, 10)
    for i in range(100):
        print(optimizer())

test_hebbbo()