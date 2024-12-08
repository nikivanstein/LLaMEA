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
        # Select two random individuals from the population
        parent1, parent2 = random.sample(self.population, 2)
        # Select a random mutation point within the search space
        mutation_point = random.randint(0, self.dim)
        # Swap the genes of the two parents
        individual = parent1[:mutation_point] + parent2[mutation_point:]
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point within the search space
        crossover_point = random.randint(0, self.dim)
        # Create a new individual by combining the genes of the two parents
        child = parent1[:crossover_point] + parent2[crossover_point:]
        # Return the new individual
        return child

class BlackBoxOptimizerMetaheuristic:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

    def __call__(self, func):
        # Initialize the population with random individuals
        self.population = [self.optimizer.__call__(func) for _ in range(self.population_size)]
        # Evaluate the fitness of each individual
        fitness = [self.optimizer.__call__(func) for func in self.population]
        # Select the fittest individuals
        self.population = [individual for _, individual in sorted(zip(fitness, self.population), reverse=True)]
        # Perform crossover and mutation
        while len(self.population) > 1:
            # Select two parents
            parent1, parent2 = random.sample(self.population, 2)
            # Perform crossover
            child = self.crossover(parent1, parent2)
            # Perform mutation
            child = self.mutate(child)
            # Replace the parents with the new individual
            self.population[0], self.population[1] = parent1, parent2
        # Return the fittest individual
        return self.population[0]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 