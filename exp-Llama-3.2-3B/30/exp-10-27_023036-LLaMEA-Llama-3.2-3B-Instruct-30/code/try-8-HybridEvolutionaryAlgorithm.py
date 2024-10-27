import numpy as np
import random
import operator

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1
        self.elite_size = 10
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
            population.append(individual)
        return population

    def fitness(self, individual):
        return func(individual)

    def evaluate(self):
        fitnesses = [self.fitness(individual) for individual in self.population]
        sorted_indices = np.argsort(fitnesses)
        self.population = [self.population[i] for i in sorted_indices]
        if len(self.population) > self.elite_size:
            self.population = self.population[:self.elite_size]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child = [0] * self.dim
            for i in range(self.dim):
                if random.random() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            return child
        else:
            return parent1

    def mutation(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, self.dim - 1)
            individual[index] += random.uniform(-1, 1)
            individual[index] = max(self.lower_bound, min(individual[index], self.upper_bound))
        return individual

    def evolve(self):
        for _ in range(self.budget):
            self.evaluate()
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(self.population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)
            self.population = new_population
            self.evaluate()

    def __call__(self, func):
        self.evolve()
        return self.population[np.argmax([self.fitness(individual) for individual in self.population])]

# Example usage:
hybrid_ea = HybridEvolutionaryAlgorithm(100, 10)
best_solution = hybrid_ea(func)
print(f"Best solution: {best_solution}")