import numpy as np
import random
from scipy import optimize

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.pso_alpha = 0.8
        self.pso_beta = 0.4
        self.pso_gamma = 0.4
        self.ga_crossover_prob = 0.9
        self.ga_mutation_prob = 0.1

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evolve the population
        for _ in range(self.budget):
            # Evaluate the population
            fitness = [func(x) for x in population]

            # Select the fittest individuals
            fittest_indices = np.argsort(fitness)[:self.population_size//2]
            fittest_individuals = population[fittest_indices]

            # Create a new population using Differential Evolution
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = parent1 + (parent2 - parent1) * np.random.uniform(-1.0, 1.0, self.dim)
                new_population.append(child)

            # Create a new population using Particle Swarm Optimization
            new_population_pso = []
            for _ in range(self.population_size):
                particle = np.random.uniform(-5.0, 5.0, self.dim)
                velocity = np.zeros(self.dim)
                best_position = particle
                best_fitness = func(particle)
                for _ in range(self.budget//2):
                    new_particle = particle + velocity
                    new_fitness = func(new_particle)
                    if new_fitness < best_fitness:
                        best_position = new_particle
                        best_fitness = new_fitness
                    velocity += self.pso_alpha * (best_position - particle) + self.pso_beta * np.random.uniform(-1.0, 1.0, self.dim)
                new_population_pso.append(best_position)

            # Create a new population using Genetic Algorithm
            new_population_ga = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = parent1 + parent2 * np.random.uniform(0.0, 1.0, self.dim)
                new_population_ga.append(child)

            # Combine the new populations
            population = np.array(new_population + new_population_pso + new_population_ga)

            # Apply mutation and crossover
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    population[i] += np.random.uniform(-1.0, 1.0, self.dim)
                if np.random.rand() < self.crossover_rate:
                    parent1, parent2 = random.sample(fittest_individuals, 2)
                    child = parent1 + parent2 * np.random.uniform(0.0, 1.0, self.dim)
                    population[i] = child

        # Return the fittest individual
        fitness = [func(x) for x in population]
        fittest_index = np.argsort(fitness)[-1]
        return population[fittest_index]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
best_individual = algorithm(func)
print(best_individual)