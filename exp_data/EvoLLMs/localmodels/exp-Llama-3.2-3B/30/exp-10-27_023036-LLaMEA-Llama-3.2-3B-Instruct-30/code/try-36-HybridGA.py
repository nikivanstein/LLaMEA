import random
import numpy as np

class HybridGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        # Initialize the population
        population = [random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        # Evaluate the initial population
        fitnesses = [func(individual) for individual in population]
        best_individual = np.argmin(fitnesses)
        best_individual = population[best_individual]

        # Evolve the population
        for _ in range(self.budget):
            # Select parents
            parents = random.sample(population, self.population_size)

            # Perform crossover
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = [(parent1[i] + parent2[i]) / 2 for i in range(self.dim)]
                if random.random() < self.crossover_rate:
                    child = random.sample(parents, 2)
                offspring.append(child)

            # Perform mutation
            for i in range(self.population_size):
                if random.random() < self.mutation_rate:
                    mutation = [random.uniform(-1.0, 1.0) for _ in range(self.dim)]
                    offspring[i] = [x + y for x, y in zip(offspring[i], mutation)]

            # Evaluate the offspring
            fitnesses = [func(individual) for individual in offspring]
            best_individual = np.argmin(fitnesses)
            best_individual = offspring[best_individual]

            # Replace the least fit individual
            population[best_individual] = best_individual

            # Update the best individual
            if fitnesses[best_individual] < fitnesses[best_individual - 1]:
                best_individual = population[best_individual]

# Test the algorithm
def func(individual):
    return sum(x**2 for x in individual)

ga = HybridGA(budget=100, dim=10)
best_individual = ga(func)
print(best_individual)