import numpy as np
from scipy.optimize import differential_evolution

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

    def update(self, func, budget=100, dim=5):
        # Generate a population of random solutions
        population = self.generate_population(budget, dim)
        
        # Evaluate each solution in the population
        fitnesses = [func(individual) for individual in population]
        
        # Select the fittest solutions
        fittest_individuals = np.array(population)[np.argsort(fitnesses)]
        
        # Refine the fittest individual using gradient descent
        refined_individuals = self.update_individuals(fittest_individuals, func, budget, dim)
        
        # Replace the old population with the new population
        population = refined_individuals
        
        return population

    def generate_population(self, budget, dim):
        # Generate a population of random solutions
        population = np.random.uniform(-5.0, 5.0, (budget, dim))
        
        return population

    def update_individuals(self, individuals, func, budget, dim):
        # Refine each individual using gradient descent
        learning_rate = self.learning_rate * (1 - self.explore_rate / budget)
        for i, individual in enumerate(individuals):
            # Learn a new direction using gradient descent
            dx = -np.dot(individual - func(individual), np.gradient(func(individual)))
            
            # Update the individual
            individuals[i] += learning_rate * dx
        
        return individuals

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 