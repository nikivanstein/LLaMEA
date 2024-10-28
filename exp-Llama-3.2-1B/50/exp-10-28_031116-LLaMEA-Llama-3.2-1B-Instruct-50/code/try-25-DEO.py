import random
import numpy as np

class DEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_function = self.evaluate_function

    def initialize_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def evaluate_function(self, func, solution):
        # Evaluate the fitness of a given solution
        fitness = func(solution)
        return fitness

    def __call__(self, func):
        # Optimize the black box function using the current population
        while self.budget > 0:
            # Select the fittest solutions
            fittest_solutions = sorted(self.population, key=self.fitness_function, reverse=True)[:self.population_size // 2]
            
            # Perform crossover and mutation to create new solutions
            new_solutions = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(fittest_solutions, 2)
                child = (parent1 + parent2) / 2
                if random.random() < 0.5:  # Mutation
                    child[0] += random.uniform(-1, 1)
                    child[1] += random.uniform(-1, 1)
                new_solutions.append(child)
            
            # Replace the least fit solutions with the new ones
            self.population = fittest_solutions + new_solutions
            
            # Update the fitness function with the new solutions
            self.population = self.evaluate_function(func, self.population)
            
            # Reduce the budget
            self.budget -= 1
            
            # Limit the population to the budget
            self.population = self.population[:self.budget]
        
        # Return the best solution
        return self.population[0]

# One-line description: Dynamic Evolutionary Optimization (DEO) algorithm for black box optimization

# Code: