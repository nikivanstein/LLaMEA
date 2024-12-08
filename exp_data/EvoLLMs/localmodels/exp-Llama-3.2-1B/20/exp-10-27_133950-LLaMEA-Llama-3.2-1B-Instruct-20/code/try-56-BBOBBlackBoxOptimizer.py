import numpy as np
from scipy.optimize import minimize
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def evolve(self, population_size, mutation_rate):
        # Initialize the population with random individuals
        population = [random.uniform(self.search_space) for _ in range(population_size)]

        # Evolve the population for a specified number of generations
        for _ in range(100):
            # Select the fittest individuals
            fittest = sorted(population, key=self.fitness, reverse=True)[:self.budget]

            # Create a new generation by adapting the fittest individuals
            new_generation = []
            for _ in range(population_size):
                # Select two parents at random from the fittest individuals
                parent1, parent2 = random.sample(fittest, 2)

                # Apply mutation to the parents
                mutated1 = parent1 + random.uniform(-1, 1)
                mutated2 = parent2 + random.uniform(-1, 1)

                # Crossover the mutated parents to create a new individual
                child = (parent1 + parent2) / 2

                # Add the new individual to the new generation
                new_generation.append(child)

            # Replace the old generation with the new generation
            population = new_generation

        # Return the fittest individual in the new generation
        return self.fitness(population[0])

    def fitness(self, individual):
        # Evaluate the fitness of an individual
        return np.mean(individual**2)

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update the optimizer with the selected solution
optimizer = BBOBBlackBoxOptimizer(1000, 10)
optimizer = optimizer.evolve(100, 0.1)
result = optimizer(func)
print(result)

# Print the updated population
for individual in optimizer.population:
    print(individual)