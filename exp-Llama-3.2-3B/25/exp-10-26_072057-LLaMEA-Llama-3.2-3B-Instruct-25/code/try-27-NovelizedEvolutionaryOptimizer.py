import numpy as np
import random

class NovelizedEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10
        self.adaptive_probability = 0.25

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
                    # Randomly change individual lines with adaptive probability
                    if random.random() < self.adaptive_probability:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                        # Ensure bounds are not exceeded
                        child = np.clip(child, -5.0, 5.0)
                    else:
                        child += np.random.uniform(-0.01, 0.01, self.dim)

                offspring.append(child)

            # Replace least fit individuals with new offspring
            population = np.array(offspring)

        # Return the best solution found
        return population[np.argmin(fitness)]

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = NovelizedEvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)