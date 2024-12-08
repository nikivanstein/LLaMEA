import numpy as np
import random

class SEEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.num_generations = int(budget / (self.population_size * self.dim))
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.initialize_population()
        self.mutation_prob = 0.2

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def fitness(self, x):
        return np.sum(x ** 2)

    def evaluate(self, x):
        return self.fitness(x)

    def selection(self, population):
        fitnesses = np.array([self.evaluate(x) for x in population])
        indices = np.argsort(fitnesses)
        return population[indices]

    def crossover(self, parent1, parent2):
        child = np.zeros_like(parent1)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def mutation(self, x):
        for i in range(self.dim):
            if random.random() < self.mutation_prob:
                x[i] += np.random.uniform(-1, 1)
                x[i] = np.clip(x[i], self.lower_bound, self.upper_bound)
        return x

    def enhance(self, population):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutation(child)
            new_population.append(child)
        return new_population

    def run(self, func):
        for generation in range(self.num_generations):
            population = self.evaluate(self.population)
            population = self.selection(population)
            population = self.enhance(population)
            population = self.initialize_population()
            best = min(population, key=self.evaluate)
            print(f'Generation {generation+1}, Best Fitness: {self.evaluate(best)}')
            if self.budget <= generation * self.dim:
                break
        return best

# Example usage
def func(x):
    return np.sum(x ** 2)

seea = SEEA(budget=100, dim=10)
best = seea(func)
print(f'Best Solution: {best}')