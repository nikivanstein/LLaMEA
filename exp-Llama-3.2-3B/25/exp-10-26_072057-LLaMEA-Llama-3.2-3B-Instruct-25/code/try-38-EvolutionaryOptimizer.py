import numpy as np
import random
import copy

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10
        self.probability = 0.25

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

                # Randomly change individual lines with a probability of 0.25
                if random.random() < self.probability:
                    child = self.change_individual_lines(child)

                offspring.append(child)

            # Replace least fit individuals with new offspring
            population = np.array(offspring)

        # Return the best solution found
        return population[np.argmin(fitness)]

    def change_individual_lines(self, individual):
        # Select a random line to change
        line_to_change = random.randint(0, self.dim-1)
        
        # Generate a new value for the line
        new_value = np.random.uniform(-5.0, 5.0)
        
        # Create a copy of the individual
        new_individual = copy.deepcopy(individual)
        
        # Change the line
        new_individual[line_to_change] = new_value
        
        return new_individual

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)