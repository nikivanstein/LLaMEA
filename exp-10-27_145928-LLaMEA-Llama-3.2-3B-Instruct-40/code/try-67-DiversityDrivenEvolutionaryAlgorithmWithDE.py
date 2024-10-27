import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithmWithDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.population = self.initialize_population()
        self.differential_evolution = self.initialize_differential_evolution()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def initialize_differential_evolution(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        parents = self.select_parents(fitnesses)
        offspring = self.crossover(parents)
        offspring = self.mutate(offspring)
        self.population = np.concatenate((parents, offspring))
        self.population = self.select_parents(fitnesses)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, parents):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def differential_evolution(self, population):
        new_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < 0.4:
                parent1 = random.choice(new_population)
                parent2 = random.choice(new_population)
                new_population[i] = (parent1 + parent2) / 2
        return new_population

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