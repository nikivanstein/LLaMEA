import numpy as np
import random

class SwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.harmony_memory_size = 100
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            population.append(individual)
        return population

    def fitness(self, func, individual):
        return func(individual)

    def evaluate_population(self, func):
        fitnesses = [self.fitness(func, individual) for individual in self.population]
        return fitnesses

    def select_parents(self, fitnesses):
        parents = np.array(self.population)[np.argsort(fitnesses)]
        return parents[:self.population_size//2]

    def crossover(self, parents):
        offspring = []
        for _ in range(self.population_size//2):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        mutated_offspring = []
        for individual in offspring:
            for i in range(self.dim):
                if random.random() < 0.1:
                    individual[i] += random.uniform(-1, 1)
                    individual[i] = max(self.lower_bound, min(individual[i], self.upper_bound))
            mutated_offspring.append(individual)
        return mutated_offspring

    def harmony_search(self, func):
        for _ in range(self.budget):
            fitnesses = self.evaluate_population(func)
            parents = self.select_parents(fitnesses)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            self.population = np.array(offspring)
            new_fitnesses = self.evaluate_population(func)
            for i, (parent, new_individual) in enumerate(zip(parents, offspring)):
                if new_fitnesses[i] < fitnesses[i]:
                    self.population[i] = new_individual

    def __call__(self, func):
        self.harmony_search(func)
        return self.population[np.argmin([self.fitness(func, individual) for individual in self.population])]

# Example usage:
def func(x):
    return sum(i**2 for i in x)

budget = 100
dim = 10
swarm_harmony_search = SwarmHarmonySearch(budget, dim)
best_individual = swarm_harmony_search(func)
print(best_individual)