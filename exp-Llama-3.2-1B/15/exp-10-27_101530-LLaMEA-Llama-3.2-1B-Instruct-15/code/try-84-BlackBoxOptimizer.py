import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Randomly select a dimension to mutate
        dim_to_mutate = random.randint(0, self.dim - 1)
        # Generate a new value for the selected dimension
        new_value = random.uniform(self.search_space[0], self.search_space[1])
        # Update the individual with the new value
        individual[dim_to_mutate] = new_value
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Randomly select a crossover point
        crossover_point = random.randint(0, self.dim - 1)
        # Create a new child individual by combining the parents
        child = [parent1[i] for i in range(crossover_point)] + [parent2[i] for i in range(crossover_point, self.dim)]
        # Return the child individual
        return child

    def evolve(self, population_size, mutation_rate, crossover_rate):
        # Initialize the population with random individuals
        population = [[random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim)] for _ in range(population_size)]
        # Evolve the population for the specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual in the population
            fitnesses = [self.__call__(func) for func in population]
            # Select the fittest individuals
            fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]
            # Create new individuals by mutating and crossover-ing the fittest individuals
            new_population = [self.mutate(fittest_individual) for _ in range(population_size)]
            # Create new individuals by crossover-ing two fittest individuals
            new_population.extend([self.crossover(fittest1, fittest2) for fittest1, fittest2 in zip(fittest_individuals, fittest_individuals[1:])])
            # Replace the old population with the new population
            population = new_population
        # Return the best individual in the population
        return self.search_space[0], self.search_space[1]

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# This algorithm uses a combination of mutation, crossover, and selection to optimize black box functions.

# Code
# ```python
def __init__(self, budget, dim):
    self.budget = budget
    self.dim = dim
    self.search_space = (-5.0, 5.0)
    self.func_evaluations = 0

def __call__(self, func):
    while self.func_evaluations < self.budget:
        # Generate a random point in the search space
        point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        # Evaluate the function at the point
        func_value = func(point)
        # Increment the function evaluations
        self.func_evaluations += 1
        # Check if the point is within the budget
        if self.func_evaluations < self.budget:
            # If not, return the point
            return point
    # If the budget is reached, return the best point found so far
    return self.search_space[0], self.search_space[1]

def mutate(individual):
    dim_to_mutate = random.randint(0, individual[-1] - 1)
    new_value = random.uniform(individual[0], individual[0] + 5)
    individual[dim_to_mutate] = new_value
    return individual

def crossover(parent1, parent2):
    crossover_point = random.randint(0, parent1[-1] - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def evolve(population_size, mutation_rate, crossover_rate):
    population = [[random.uniform(self.search_space[0], self.search_space[1]) for _ in range(population_size)] for _ in range(population_size)]
    for _ in range(100):
        fitnesses = [self.__call__(func) for func in population]
        fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]
        new_population = [mutate(fittest_individual) for fittest_individual in fittest_individuals[:int(population_size/2)]]
        new_population.extend([crossover(mutate(fittest_individual), mutate(fittest_individual)) for fittest_individual in fittest_individuals[int(population_size/2):]])
        population = new_population
    return population[0], population[1]

# Example usage
budget = 1000
dim = 5
best_individual, best_fitness = evolve(population_size=100, mutation_rate=0.1, crossover_rate=0.5)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)