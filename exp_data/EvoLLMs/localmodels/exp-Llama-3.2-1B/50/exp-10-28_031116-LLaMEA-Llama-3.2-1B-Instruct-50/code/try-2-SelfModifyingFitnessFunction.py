import random
import numpy as np

class SelfModifyingFitnessFunction:
    def __init__(self, budget, dim, func, mutation_prob=0.1, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.func = func
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.population = self.generate_population()

    def __call__(self, func):
        for _ in range(self.budget):
            func = self.func(func)
        return func

    def generate_population(self):
        population = []
        for _ in range(1000):
            dim = self.dim + random.randint(-5, 5)
            solution = np.random.uniform(-5.0, 5.0, dim)
            fitness = self.func(solution)
            if random.random() < self.mutation_prob:
                dim = self.dim + random.randint(-5, 5)
                solution = np.random.uniform(-5.0, 5.0, dim)
                fitness = self.func(solution)
            population.append((solution, fitness))
        return population

    def select(self, population, dim):
        selection_prob = 0.5
        indices = random.choices(range(len(population)), weights=population[1], k=len(population))
        selected_indices = [i for i, _ in indices]
        selected_population = [population[i][0] for i in selected_indices]
        return selected_population

    def crossover(self, parent1, parent2):
        dim = self.dim + random.randint(-5, 5)
        child = np.zeros(dim)
        for i in range(dim):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def mutate(self, solution):
        if random.random() < self.mutation_rate:
            dim = self.dim + random.randint(-5, 5)
            mutation = np.random.uniform(-5.0, 5.0, dim)
            solution += mutation
            solution = np.clip(solution, -5.0, 5.0)
        return solution

    def __str__(self):
        return f"Fitness: {self.func(np.array([self.population[0][0]]).reshape(-1, self.dim))}"

# Description: Evolutionary Optimization using Self-Modifying Fitness Functions
# Code: 
# ```python
def func(x):
    return np.sum(x**2)

self_modifying_fitness_function = SelfModifyingFitnessFunction(budget=100, dim=10, func=func)
print(self_modifying_fitness_function)