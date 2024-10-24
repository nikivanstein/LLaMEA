import numpy as np
from scipy.optimize import minimize
from collections import deque

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = deque()
        self.best_solution = None
        self.best_score = float('-inf')
        self.search_space = 2.5  # range: [-5.0, 5.0]

    def __call__(self, func):
        # Evaluate the function using self.budget function evaluations
        func_evaluations = np.random.randint(0, self.budget + 1)
        func_value = func(func_evaluations)
        
        # Select a random individual from the search space
        if func_value > 0:
            individual = np.random.uniform(-self.search_space, self.search_space, size=self.dim)
        else:
            individual = np.random.uniform(-self.search_space, self.search_space, size=self.dim)
        
        # Generate an initial population of random solutions
        for _ in range(100):
            solution = np.random.uniform(-self.search_space, self.search_space, size=self.dim)
            population.append(solution)
        
        # Evolve the population using the HybridMetaheuristic algorithm
        while len(population) > 0:
            # Select the fittest individual
            fittest_individual = population[np.argmax([self.evaluate_func(individual) for individual in population])]
            
            # Generate a new individual by perturbing the fittest individual
            new_individual = fittest_individual + np.random.normal(0, 1, size=self.dim)
            
            # Check if the new individual is within the search space
            if np.all(new_individual >= -self.search_space) and np.all(new_individual <= self.search_space):
                # Evaluate the new individual using self.budget function evaluations
                func_evaluations = np.random.randint(0, self.budget + 1)
                func_value = self.evaluate_func(new_individual)
                
                # Check if the new individual is better than the current best solution
                if func_value < self.best_score:
                    # Update the best solution and score
                    self.best_solution = new_individual
                    self.best_score = func_value
                    self.population.append(new_individual)
                else:
                    # Refine the search space by perturbing the new individual
                    new_individual = fittest_individual + np.random.normal(0, 1, size=self.dim)
                    while np.all(new_individual >= -self.search_space) and np.all(new_individual <= self.search_space):
                        func_evaluations = np.random.randint(0, self.budget + 1)
                        func_value = self.evaluate_func(new_individual)
                        
                        # Check if the new individual is better than the current best solution
                        if func_value < self.best_score:
                            # Update the best solution and score
                            self.best_solution = new_individual
                            self.best_score = func_value
                            self.population.append(new_individual)
                            break
                        else:
                            # Refine the search space by perturbing the new individual
                            new_individual = fittest_individual + np.random.normal(0, 1, size=self.dim)
                    # If no improvement is found, discard the new individual
                    else:
                        self.population.pop()
        
        # Return the best solution found
        return self.best_solution

    def evaluate_func(self, individual):
        # Evaluate the black box function using the given individual
        return individual[0]

# Test the algorithm
func = lambda x: x**2
hybrid_metaheuristic = HybridMetaheuristic(100, 2)
best_solution = hybrid_metaheuristic(func)
print("Best solution:", best_solution)
print("Best score:", hybrid_metaheuristic.evaluate_func(best_solution))