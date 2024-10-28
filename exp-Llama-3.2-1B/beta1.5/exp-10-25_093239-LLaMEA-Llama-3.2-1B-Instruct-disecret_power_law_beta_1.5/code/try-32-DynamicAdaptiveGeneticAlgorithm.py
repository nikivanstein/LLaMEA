import random
import math
import numpy as np

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func):
        # Initialize the population with random functions
        self.population = [func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)]

        # Define the selection strategy based on the fitness and the dimension
        def selection_strategy(fitness, dim):
            if fitness > 0:
                return random.choices(range(self.population_size), weights=[1 / fitness for _ in range(self.population_size)], k=1)[0]
            else:
                return random.choices(range(self.population_size), weights=[1 / -fitness for _ in range(self.population_size)], k=1)[0]

        # Define the crossover strategy
        def crossover(parent1, parent2):
            # Select the parent with the highest fitness
            parent1, parent2 = max((selection_strategy(f, dim) for f in [self.fitnesses[self.population_size - 1]]), key=lambda x: x), max((selection_strategy(f, dim) for f in [self.fitnesses[self.population_size - 1]]), key=lambda x: x))

            # Perform crossover
            child1 = parent1[:dim] + parent2[dim:]
            child2 = parent2[:dim] + parent1[dim:]

            return child1, child2

        # Define the mutation strategy
        def mutation(individual):
            # Select a random individual to mutate
            parent1, parent2 = random.sample([individual], 2)

            # Perform mutation
            mutated_individual = individual[:dim] + [random.uniform(-5.0, 5.0)] + individual[dim:]

            return mutated_individual

        # Run the algorithm for the specified number of generations
        for _ in range(self.budget):
            # Select the next individuals using the selection strategy
            next_individuals = [selection_strategy(f, dim) for f in self.fitnesses]

            # Perform crossover and mutation
            next_individuals = [crossover(individual, individual) for individual in next_individuals]
            next_individuals = [mutation(individual) for individual in next_individuals]

            # Update the population
            self.population = next_individuals

            # Ensure the fitness stays within the bounds
            self.fitnesses = [min(max(f, -5.0), 5.0) for f in self.fitnesses]

        # Return the best individual
        return self.population[0]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.