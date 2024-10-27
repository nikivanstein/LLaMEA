import numpy as np
import random

class PEABBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = self.get_best_solution()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def get_best_solution(self):
        best_solution = min(self.population, key=lambda x: self.evaluate_function(x))
        return best_solution

    def evaluate_function(self, solution):
        return self.func(solution)

    def update_population(self, func):
        for _ in range(self.population_size):
            if self.budget > 0:
                # Select a random parent from the current population
                parent = random.choice(self.population)
                # Generate a child using crossover and mutation
                child = self.crossover(parent)
                child = self.mutate(child)
                # Evaluate the child and replace the worst solution in the population
                child_score = self.evaluate_function(child)
                if child_score < self.evaluate_function(min(self.population, key=lambda x: self.evaluate_function(x))):
                    self.population.remove(min(self.population, key=lambda x: self.evaluate_function(x)))
                    self.population.append(child)
                    self.best_solution = min(self.population, key=lambda x: self.evaluate_function(x))
                    self.budget -= 1
            else:
                break

    def crossover(self, parent):
        child = parent.copy()
        # Perform a random crossover
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent[i] + np.random.uniform(-1.0, 1.0)
        return child

    def mutate(self, solution):
        # Perform a random mutation
        for i in range(self.dim):
            if random.random() < 0.1:
                solution[i] += np.random.uniform(-1.0, 1.0)
        return solution

    def __call__(self, func):
        for _ in range(self.budget):
            self.update_population(func)
        return self.best_solution

# Example usage:
def func(x):
    return np.sum(x**2)

peabbo = PEABBO(budget=100, dim=10)
best_solution = peabbo(func)
print(best_solution)