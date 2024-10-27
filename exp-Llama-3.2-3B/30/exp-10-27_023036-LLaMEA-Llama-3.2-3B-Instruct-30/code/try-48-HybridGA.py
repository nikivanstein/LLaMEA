import random
import math
import numpy as np

class HybridGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.temperature = 1000
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)

        # Evaluate initial population
        scores = [func(individual) for individual in population]
        for i in range(self.population_size):
            population[i] = (population[i], scores[i])

        # Genetic Algorithm
        for _ in range(self.budget):
            # Select parents
            parents = random.sample(population, self.population_size // 2)

            # Crossover
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(parents, 2)
                child = [(parent1[0][i] + parent2[0][i]) / 2 for i in range(self.dim)]
                offspring.append(child)

            # Mutation
            for i in range(self.population_size // 2):
                if random.random() < self.mutation_rate:
                    mutation = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
                    child = [(child[0][i] + mutation[i]) for i in range(self.dim)]
                    offspring[i] = (child, [func(child)])

            # Update population
            population = offspring + parents

            # Simulated Annealing
            scores = [func(individual[0]) for individual in population]
            for i in range(self.population_size):
                if scores[i] < population[i][1]:
                    population[i] = (population[i][0], scores[i])
                elif random.random() < math.exp((population[i][1] - scores[i]) / self.temperature):
                    population[i] = (population[i][0], scores[i])

            # Cool down temperature
            self.temperature *= self.cooling_rate

        # Return best individual
        best_individual = max(population, key=lambda individual: individual[1])
        return best_individual[0]

# Test the algorithm
def func(x):
    return sum(i**2 for i in x)

hybrid_ga = HybridGA(budget=100, dim=10)
best_individual = hybrid_ga(func)
print(best_individual)