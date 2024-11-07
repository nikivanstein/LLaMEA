import numpy as np
import random

class SwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.wavelength = 0.5
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.harmony_memory = []
        self.best_solution = None
        self.population = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        if self.best_solution is None:
            self.best_solution = np.array([np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)])
            self.harmony_memory.append(self.best_solution.copy())
        
        for _ in range(self.budget):
            # Initialize population with random solutions
            self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            self.population = self.filter_solutions(self.population, func)
            self.harmony_memory.append(self.population.copy())
            self.update_best_solution(self.population)

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(self.dim):
            if random.random() < self.wavelength:
                child[i] = (parent1[i] + parent2[i]) / 2
        return child

    def mutate(self, solution):
        if random.random() < self.mutation_probability:
            index = random.randint(0, self.dim - 1)
            solution[index] += np.random.uniform(-1.0, 1.0)
        return solution

    def filter_solutions(self, solutions, func):
        fitness = func(solutions)
        best_idx = np.argmin(fitness)
        return solutions[best_idx]

    def update_best_solution(self, solutions):
        fitness = func(solutions)
        best_idx = np.argmin(fitness)
        if np.all(fitness[best_idx] < func(self.best_solution)):
            self.best_solution = solutions[best_idx]

# Example usage
def func(x):
    return np.sum(x**2)

swarm = SwarmHarmonySearch(budget=100, dim=10)
swarm('func')