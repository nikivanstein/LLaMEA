# Description: Novel Metaheuristic Algorithm for Black Box Optimization (Metaheuristic)
# Code: 
# ```python
import random
import numpy as np
import math
import copy

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        # Refine the strategy based on the number of evaluations
        if num_evals < 10:
            # Increase the mutation rate
            self.mutation_rate = 0.1
        elif num_evals < 20:
            # Increase the exploration rate
            self.exploration_rate = 0.05
        else:
            # Decrease the mutation rate
            self.mutation_rate = 0.01
            # Decrease the exploration rate
            self.exploration_rate = 0.005

        return best_func

def bbo(func, budget, dim, mutation_rate, exploration_rate, num_evals):
    # Initialize the population with random individuals
    population = [copy.deepcopy(func) for _ in range(100)]

    # Evolve the population over time
    for _ in range(budget):
        # Evaluate the fitness of each individual
        fitnesses = [func(individual) for individual in population]

        # Select the fittest individuals
        fittest_individuals = sorted(zip(fitnesses, population), reverse=True)[:100]

        # Select the best individual to reproduce
        parent1, parent2 = random.sample(fittest_individuals, 2)

        # Perform crossover and mutation
        child1 = parent1[:dim // 2] + [func(x) for x in random.sample(parent1[dim // 2:], dim // 2)] + [parent1[dim // 2 + dim // 2]]
        child2 = parent2[:dim // 2] + [func(x) for x in random.sample(parent2[dim // 2:], dim // 2)] + [parent2[dim // 2 + dim // 2]]

        # Apply mutation
        if random.random() < mutation_rate:
            child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

        if random.random() < exploration_rate:
            if random.randint(0, dim - 1) not in [child1.index(x) for x in child1]:
                child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

        # Replace the parent individuals with the child
        population[0] = child1
        population[1] = child2

    # Evaluate the fitness of each individual
    fitnesses = [func(individual) for individual in population]

    # Select the fittest individual to reproduce
    fittest_individuals = sorted(zip(fitnesses, population), reverse=True)[:100]

    # Select the best individual to reproduce
    parent1, parent2 = random.sample(fittest_individuals, 2)

    # Perform crossover and mutation
    child1 = parent1[:dim // 2] + [func(x) for x in random.sample(parent1[dim // 2:], dim // 2)] + [parent1[dim // 2 + dim // 2]]
    child2 = parent2[:dim // 2] + [func(x) for x in random.sample(parent2[dim // 2:], dim // 2)] + [parent2[dim // 2 + dim // 2]]

    # Apply mutation
    if random.random() < mutation_rate:
        child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

    if random.random() < exploration_rate:
        if random.randint(0, dim - 1) not in [child1.index(x) for x in child1]:
            child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

    # Replace the parent individuals with the child
    population[0] = child1
    population[1] = child2

    return population

# Test the algorithm
def bbo_test():
    # Define the function to optimize
    def func(x):
        return x**2 + 2*x + 1

    # Define the parameters
    budget = 100
    dim = 2
    mutation_rate = 0.01
    exploration_rate = 0.005

    # Evolve the population
    population = Metaheuristic(budget, dim).__call__(func)

    # Evaluate the fitness of each individual
    fitnesses = [func(individual) for individual in population]

    # Select the fittest individual to reproduce
    fittest_individuals = sorted(zip(fitnesses, population), reverse=True)[:100]

    # Select the best individual to reproduce
    parent1, parent2 = random.sample(fittest_individuals, 2)

    # Perform crossover and mutation
    child1 = parent1[:dim // 2] + [func(x) for x in random.sample(parent1[dim // 2:], dim // 2)] + [parent1[dim // 2 + dim // 2]]
    child2 = parent2[:dim // 2] + [func(x) for x in random.sample(parent2[dim // 2:], dim // 2)] + [parent2[dim // 2 + dim // 2]]

    # Apply mutation
    if random.random() < mutation_rate:
        child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

    if random.random() < exploration_rate:
        if random.randint(0, dim - 1) not in [child1.index(x) for x in child1]:
            child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

    # Replace the parent individuals with the child
    population[0] = child1
    population[1] = child2

    # Evaluate the fitness of each individual
    fitnesses = [func(individual) for individual in population]

    # Select the fittest individual to reproduce
    fittest_individuals = sorted(zip(fitnesses, population), reverse=True)[:100]

    # Select the best individual to reproduce
    parent1, parent2 = random.sample(fittest_individuals, 2)

    # Perform crossover and mutation
    child1 = parent1[:dim // 2] + [func(x) for x in random.sample(parent1[dim // 2:], dim // 2)] + [parent1[dim // 2 + dim // 2]]
    child2 = parent2[:dim // 2] + [func(x) for x in random.sample(parent2[dim // 2:], dim // 2)] + [parent2[dim // 2 + dim // 2]]

    # Apply mutation
    if random.random() < mutation_rate:
        child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

    if random.random() < exploration_rate:
        if random.randint(0, dim - 1) not in [child1.index(x) for x in child1]:
            child1[random.randint(0, dim - 1)] = func(random.randint(0, dim - 1))

    # Replace the parent individuals with the child
    population[0] = child1
    population[1] = child2

    # Return the best individual
    return max(set(population, key=population.count), key=population.count)

# Run the test
best_individual = bbo_test()
print("The best individual is:", best_individual)