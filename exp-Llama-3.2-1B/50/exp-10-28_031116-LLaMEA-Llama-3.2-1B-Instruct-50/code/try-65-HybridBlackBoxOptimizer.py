import numpy as np
import random
from scipy.optimize import minimize

class HybridBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.population_history = []
        self.best_solution = None
        self.best_score = -np.inf

    def __call__(self, func):
        # Evaluate the black box function
        func_values = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
        
        # Select a random subset of function evaluations
        subset_idx = random.sample(range(self.budget), self.budget)
        
        # Initialize the population
        self.population = []
        for _ in range(self.budget):
            # Randomly select a subset of function evaluations
            subset_idx = random.sample(subset_idx, self.dim)
            
            # Add the subset to the population
            self.population.append(func_values[subset_idx])
        
        # Perform the hybrid algorithm
        for _ in range(100):  # Number of iterations
            # Refine the population using a novel heuristic
            refined_population = self.refine_population()
            
            # Evaluate the refined population
            refined_func_values = np.array([func(x) for x in refined_population])
            
            # Select the fittest individual
            fittest_idx = np.argmin(refined_func_values)
            
            # Replace the worst individual with the fittest one
            worst_idx = subset_idx[fittest_idx]
            worst_func_value = refined_func_values[worst_idx]
            self.population[worst_idx] = func_values[worst_idx]
            func_values[worst_idx] = worst_func_value
            
            # Add the new individual to the population
            self.population.append(func_values[fittest_idx])
        
        # Evaluate the final population
        final_func_values = np.array([func(x) for x in self.population])
        
        # Update the best solution and score
        self.best_solution = np.min(final_func_values)
        self.best_score = np.min(final_func_values)
        
        # Update the population history
        self.population_history.append(final_func_values)
        self.population_history.append(np.max(final_func_values))
        
        # Update the best score
        self.best_score = np.min(final_func_values)
        
        # Update the best solution
        self.best_solution = np.min(final_func_values)
        
        return self.best_solution
    
    def refine_population(self):
        # Novel heuristic: Select the fittest individual based on the probability of refinement
        # and the probability of rejection
        fittest_idx = np.argmin(np.array(self.population))
        prob_refine = random.random()
        prob_reject = 1 - prob_refine
        
        if prob_refine > 0.45:  # Refine the fittest individual
            return self.population[:fittest_idx]
        else:  # Reject the fittest individual
            return np.delete(self.population, fittest_idx)