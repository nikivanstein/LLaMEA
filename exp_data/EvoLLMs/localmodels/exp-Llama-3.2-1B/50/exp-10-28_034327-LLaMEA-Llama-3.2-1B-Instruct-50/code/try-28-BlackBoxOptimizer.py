import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation(self, func, budget):
        # Initialize the population with random points
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
        
        # Run the selection and crossover steps for 100 generations
        for _ in range(100):
            # Select parents using tournament selection
            parents = sorted([self.evaluate_fitness(parent) for parent in population])
            parent1, parent2 = parents[:int(0.5 * len(parents))]
            
            # Perform crossover to create offspring
            offspring = []
            while len(offspring) < 10:
                # Select a random parent
                parent = random.choice([parent1, parent2])
                # Perform crossover (iterated permutation)
                child = self.iterated_permutation(func, 100)
                # Add the child to the offspring list
                offspring.append(child)
            
            # Replace the least fit individuals with the new offspring
            population = [offspring[i] for i in range(len(population))]
        
        # Select the fittest individual
        fittest_individual = population[0]
        
        # Return the fittest individual as the best point found so far
        return fittest_individual

    def iterated_cooling(self, func, budget):
        # Initialize the population with random points
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
        
        # Run the selection and mutation steps for 100 generations
        for _ in range(100):
            # Select parents using tournament selection
            parents = sorted([self.evaluate_fitness(parent) for parent in population])
            parent1, parent2 = parents[:int(0.5 * len(parents))]
            
            # Perform crossover to create offspring
            offspring = []
            while len(offspring) < 10:
                # Select a random parent
                parent = random.choice([parent1, parent2])
                # Perform crossover (iterated permutation)
                child = self.iterated_permutation(func, 100)
                # Add the child to the offspring list
                offspring.append(child)
            
            # Replace the least fit individuals with the new offspring
            population = [offspring[i] for i in range(len(population))]
        
        # Select the fittest individual
        fittest_individual = population[0]
        
        # Return the fittest individual as the best point found so far
        return fittest_individual

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 