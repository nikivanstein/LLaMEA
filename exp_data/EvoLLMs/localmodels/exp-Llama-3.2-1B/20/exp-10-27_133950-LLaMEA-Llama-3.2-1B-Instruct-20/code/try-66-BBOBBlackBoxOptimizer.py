import numpy as np
from scipy.optimize import minimize
import random
import copy

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 50
        self.population = self.initialize_population()

    def initialize_population(self):
        return [copy.deepcopy(self.search_space) for _ in range(self.population_size)]

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def mutate(self, individual):
        if random.random() < 0.2:
            self.search_space[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
        return copy.deepcopy(individual)

    def crossover(self, parent1, parent2):
        if random.random() < 0.2:
            child = copy.deepcopy(parent1)
            child[-1] = random.uniform(-5.0, 5.0)
            return child
        else:
            child = copy.deepcopy(parent2)
            return child

    def evolve(self, population):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def update_population(self, population):
        self.population = self.evolve(population)

    def evaluate_fitness(self, func, population):
        results = []
        for individual in population:
            result = func(individual)
            results.append(result)
        return results

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
results = optimizer(func)
print(results)

# Update population using the selected solution
optimizer.update_population(optimizer.population)

# Evaluate fitness using the updated population
fitness_results = optimizer.evaluate_fitness(func, optimizer.population)
print(fitness_results)

# One-line description with the main idea
# Novel Black Box Optimization using Evolutionary Strategies