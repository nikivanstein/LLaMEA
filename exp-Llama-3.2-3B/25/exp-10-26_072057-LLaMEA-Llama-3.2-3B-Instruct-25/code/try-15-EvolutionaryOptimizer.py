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
        self.adaptive_mutation_rate = 0.25

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
                    mutation = np.random.uniform(-0.1, 0.1, self.dim)
                    mutated_child = child + mutation
                    mutated_child = np.clip(mutated_child, -5.0, 5.0)  # Clip to bounds
                    mutated_child = mutated_child * np.random.choice([-1, 1], size=self.dim)  # Randomize mutation
                    offspring.append(mutated_child)

            # Replace least fit individuals with new offspring
            population = np.array(offspring)

        # Update adaptive mutation rate
        self.mutation_rate = self.mutation_rate * (1 - self.adaptive_mutation_rate)

        # Return the best solution found
        return population[np.argmin(fitness)]

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)