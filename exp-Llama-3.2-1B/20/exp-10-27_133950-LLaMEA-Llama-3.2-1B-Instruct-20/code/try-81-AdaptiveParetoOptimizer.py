import numpy as np
import random
from scipy.optimize import minimize

class AdaptiveParetoOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutate_rate = 0.01
        self.pareto_threshold = 0.2

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

    def generate_population(self):
        return [random.uniform(self.search_space) for _ in range(self.population_size)]

    def select_parents(self, population):
        return random.sample(population, int(self.population_size * self.pareto_threshold))

    def crossover(self, parent1, parent2):
        child = [x for x in parent1 if x <= random.uniform(0, 1)]
        child.extend([x for x in parent2 if x <= random.uniform(0, 1)])
        return child

    def mutate(self, individual):
        if random.random() < self.mutate_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def evaluate_fitness(self, individual, func):
        return func(individual)

    def update_population(self, population):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

# Example usage:
optimizer = AdaptiveParetoOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update population
population = optimizer.generate_population()
new_population = optimizer.update_population(population)

# Evaluate fitness
fitness = [optimizer.evaluate_fitness(individual, func) for individual in new_population]
print(fitness)