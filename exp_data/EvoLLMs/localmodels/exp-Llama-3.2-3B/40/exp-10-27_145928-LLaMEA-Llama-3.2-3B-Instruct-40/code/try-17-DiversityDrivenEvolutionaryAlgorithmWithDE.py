import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithmWithDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.DE_probability = 0.4
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                if random.random() < self.DE_probability:
                    offspring[i] = self.differential_evolution(parent1, parent2)
                else:
                    offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def differential_evolution(self, parent1, parent2):
        x1, x2 = parent1, parent2
        y1, y2 = np.random.uniform(-5.0, 5.0, self.dim), np.random.uniform(-5.0, 5.0, self.dim)
        y1 = y1 + y2
        y2 = y2 + y1
        return (x1 + y1) / 2 + (x2 + y2) / 2

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

ddea_de = DiversityDrivenEvolutionaryAlgorithmWithDE(budget=100, dim=10)
optimal_solution = ddea_de(func)
print(optimal_solution)