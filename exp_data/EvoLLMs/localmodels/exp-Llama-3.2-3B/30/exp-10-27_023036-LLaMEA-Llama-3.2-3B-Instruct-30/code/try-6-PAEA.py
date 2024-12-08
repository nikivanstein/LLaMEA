import numpy as np
import random

class PAEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def evaluate(self, func):
        self.fitnesses = [func(individual) for individual in self.population]
        self.population = self.select_population()
        self.mutate()
        self.crossover()

    def select_population(self):
        fitnesses = np.array(self.fitnesses)
        probabilities = fitnesses / np.sum(fitnesses)
        selected_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities, replace=False)
        selected_population = [self.population[i] for i in selected_indices]
        return selected_population

    def mutate(self):
        for i in range(self.population_size):
            if random.random() < self.mutation_rate:
                index = random.randint(0, self.dim - 1)
                self.population[i][index] += random.uniform(-1.0, 1.0)

    def crossover(self):
        for i in range(self.population_size):
            if random.random() < self.crossover_rate:
                parent1 = self.population[i]
                parent2 = random.choice(self.population)
                child1 = parent1[:self.dim//2] + [random.uniform(-5.0, 5.0) for _ in range(self.dim//2)]
                child2 = parent2[:self.dim//2] + [random.uniform(-5.0, 5.0) for _ in range(self.dim//2)]
                self.population[i] = child1
                self.population[i + self.population_size // 2] = child2

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return self.population[np.argmin(self.fitnesses)]

# Example usage:
def func(x):
    return np.sum(x**2)

paea = PAEA(budget=100, dim=10)
best_individual = paea(func)
print(best_individual)