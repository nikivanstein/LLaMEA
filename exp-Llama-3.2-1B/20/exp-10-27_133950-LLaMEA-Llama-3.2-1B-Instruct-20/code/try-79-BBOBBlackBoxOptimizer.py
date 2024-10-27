import numpy as np
from scipy.optimize import minimize
from collections import deque
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100
        self.population_size_deque = deque(maxlen=1000)
        self.population_debounce = 100

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

    def select_next_generation(self, population):
        next_generation = []
        for _ in range(self.population_size_deque[0]):
            new_individual = random.choice(population)
            next_generation.append(new_individual)
            self.population_size_deque.append(new_individual)
        for i in range(self.population_size_deque[0] - self.population_size_debounce):
            next_generation.append(population[0])
        return next_generation

    def mutate(self, individual):
        if random.random() < 0.2:
            self.population_size_deque.popleft()
            self.population_size_debounce += 1
        return individual

    def __next_generation(self, population):
        return self.select_next_generation(population)

    def evaluate_fitness(self, individual, fitness):
        return fitness

    def run(self, func, population_size):
        population = self.generate_population()
        best_individual = None
        best_fitness = -np.inf
        for _ in range(self.budget):
            fitness = self.evaluate_fitness(population[0], self.evaluate_fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = population[0]
            next_generation = self.__next_generation(population)
            population = next_generation
        return best_individual, best_fitness

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
best_individual, best_fitness = optimizer(func, 100)
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")

# Novel Metaheuristic Algorithm:
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 