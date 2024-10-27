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

    def hybrid_metaheuristic(self, func, bounds, mutation_rate):
        def objective(x):
            return -func(x)
        
        # Initialize the population with random points in the search space
        population = np.random.uniform(bounds[0], bounds[1], size=(self.budget, self.dim))
        
        # Define the mutation function
        def mutate(individual):
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(0, self.dim)
                individual[idx] += np.random.uniform(-1, 1)
            return individual
        
        # Run the hybrid metaheuristic algorithm
        for _ in range(1000):
            # Evaluate the fitness of each individual in the population
            fitnesses = np.array([objective(individual) for individual in population])
            
            # Select the fittest individuals to reproduce
            parents = population[np.argsort(fitnesses)]
            
            # Create a new population by crossover and mutation
            offspring = np.concatenate((parents[:int(self.budget/2)], parents[int(self.budget/2):]))
            offspring = np.concatenate((offspring, offspring[:int(self.budget/2)]))
            offspring = np.concatenate((offspring[int(self.budget/2):], offspring[:int(self.budget/2)]))
            offspring = mutate(offspring)
            
            # Replace the old population with the new one
            population = offspring
        
        # Evaluate the fitness of the final population
        fitnesses = np.array([objective(individual) for individual in population])
        
        # Return the fittest individual
        return population[np.argsort(fitnesses)[-1]]

# Description: Hybrid Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
mgdalr = MGDALR(budget=1000, dim=10)
mgdalr.budget = 1000  # Increase the budget for better results
mgdalr.dim = 10  # Increase the dimensionality for better results
mgdalr.explore_rate = 0.2  # Increase the exploration rate for better results
mgdalr.mutation_rate = 0.01  # Increase the mutation rate for better results

result = mgdalr.hybrid_metaheuristic(func, (-5.0, 5.0), 0.1)
print("Best solution:", result)