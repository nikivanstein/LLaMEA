import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        population = []
        for _ in range(100):
            dim = self.dim
            individual = [random.uniform(-5.0, 5.0) for _ in range(dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        # Evaluate the function for the given budget
        evaluations = []
        for _ in range(self.budget):
            func_value = func(self.population)
            evaluations.append(func_value)

        # Select the best individual based on the evaluations
        best_individual = self.select_best_individual(evaluations)

        # Return the best individual
        return best_individual

    def select_best_individual(self, evaluations):
        # Use a variant of the "Nash Equilibrium" strategy
        # to select the best individual
        best_individual = evaluations.index(max(evaluations))
        return best_individual

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        for _ in range(random.randint(1, self.dim)):
            mutated_individual[_] += random.uniform(-1, 1)
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Perform crossover to create a new individual
        child = parent1.copy()
        for _ in range(random.randint(1, self.dim)):
            child[_] = random.uniform(-5.0, 5.0)
        return child

    def fitness(self, individual):
        # Evaluate the fitness of the individual
        return individual[0]**2  # Simplified example fitness function

# Example usage:
budget = 100
dim = 10
optimizer = BlackBoxOptimizer(budget, dim)

# Evaluate the function for the given budget
func = lambda x: x[0]**2
evaluations = [func(x) for x in optimizer.population]

# Select the best individual
best_individual = optimizer.__call__(func)

# Print the best individual
print("Best Individual:", best_individual)

# Mutate the best individual
mutated_individual = optimizer.mutate(best_individual)

# Perform crossover to create a new individual
crossover_individual = optimizer.crossover(best_individual, mutated_individual)

# Print the new individual
print("New Individual:", crossover_individual)

# Evaluate the new individual
new_evaluations = [crossover_individual[0]**2 for x in optimizer.population]

# Update the population
optimizer.population = [x for x in optimizer.population if x in new_evaluations]