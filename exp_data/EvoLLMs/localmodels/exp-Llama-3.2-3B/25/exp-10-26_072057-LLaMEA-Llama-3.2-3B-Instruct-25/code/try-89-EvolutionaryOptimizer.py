import numpy as np
import random

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10

    def __call__(self, func):
        # Initialize population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Select fittest individuals
            fittest = population[np.argsort(fitness)]

            # Generate new offspring
            offspring = []
            for _ in range(self.population_size):
                # Select parents
                parent1, parent2 = random.sample(fittest, 2)

                # Crossover
                child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

                # Mutate
                if random.random() < self.mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)

                offspring.append(child)

            # Replace least fit individuals with new offspring
            population = np.array(offspring)

        # Select a subset of the population to refine the strategy
        refined_population = random.sample(population, int(self.population_size * 0.25))

        # Refine the strategy using swarm intelligence
        refined_population = self.swarm_refine(refined_population)

        # Return the best solution found
        return refined_population[np.argmin(fitness)]

    def swarm_refine(self, population):
        # Initialize the swarm with the refined population
        swarm = population

        # Evolve the swarm for a fixed number of generations
        for _ in range(10):
            # Evaluate the swarm
            fitness = np.array([func(x) for x in swarm])

            # Select the fittest individuals
            fittest = swarm[np.argsort(fitness)]

            # Generate new offspring
            offspring = []
            for _ in range(len(fittest)):
                # Select parents
                parent1, parent2 = random.sample(fittest, 2)

                # Crossover
                child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

                # Mutate
                if random.random() < self.mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)

                offspring.append(child)

            # Replace the least fit individuals with new offspring
            swarm = np.array(offspring)

        # Return the refined swarm
        return swarm

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)