import random
import numpy as np

class EvolutionarySwarmOptimization:
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
        return min(self.population, key=lambda x: self.evaluate(x))

    def evaluate(self, solution):
        return self.func(solution)

    def __call__(self, func):
        for _ in range(self.budget):
            # Probability-based mutation
            if random.random() < 0.3:
                solution = self.get_neighbor(self.best_solution)
            else:
                solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Selection
            if self.evaluate(solution) < self.evaluate(self.best_solution):
                self.best_solution = solution

            # Replacement
            self.population.remove(self.get_neighbor(self.best_solution))
            self.population.append(solution)

    def get_neighbor(self, solution):
        neighbor = solution.copy()
        for i in range(self.dim):
            if random.random() < 0.1:
                neighbor[i] += random.uniform(-1.0, 1.0)
        return neighbor

# Example usage
budget = 100
dim = 10
func = lambda x: x[0]**2 + x[1]**2
optimizer = EvolutionarySwarmOptimization(budget, dim)
best_solution = optimizer(func)
print("Best solution:", best_solution)
print("Best fitness:", optimizer.evaluate(best_solution))