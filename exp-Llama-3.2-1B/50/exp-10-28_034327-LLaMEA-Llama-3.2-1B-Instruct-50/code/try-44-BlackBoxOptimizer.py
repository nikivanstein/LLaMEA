import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0

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

    def iterated_permutation(self, func, bounds, budget):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(bounds[0], bounds[1], self.dim) for _ in range(100)]
        
        # Iterate for a specified number of iterations
        for _ in range(self.iterations):
            # Select the fittest individual
            fittest_individual = population[np.argmax([func(individual) for individual in population])]
            
            # Generate a new population by iterated permutation
            new_population = [fittest_individual + np.random.uniform(-1.0, 1.0, self.dim) for individual in population]
            
            # Evaluate the new population
            new_evaluations = [func(individual) for individual in new_population]
            
            # Select the fittest new individual
            new_fittest_individual = population[np.argmax(new_evaluations)]
            
            # Update the population with the new individual and bounds
            population = new_population + [new_fittest_individual]
            population = population[:budget]
        
        # Return the fittest individual in the final population
        return population[np.argmax([func(individual) for individual in population])]

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# ```python
# ```python
# # Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# # Code: 
# # ```python
# # ```python
# class IteratedPermutationAndCoolingOptimizer(BlackBoxOptimizer):
#     def __init__(self, budget, dim):
#         super().__init__(budget, dim)
        
#     def __call__(self, func):
#         bounds = (-5.0, 5.0)
#         population = super().__call__(func, bounds, budget)
        
#         # Select the fittest individual using iterated permutation
#         # with cooling
#         bounds = (-5.0, 5.0)
#         population = self.iterated_permutation(func, bounds, budget)
        
#         # Return the fittest individual in the final population
#         return population[np.argmax([func(individual) for individual in population])]

# # Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# # Code: 
# # ```python
# # ```python
# # ```python
# # ```python
# # Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# # Code: 
# # ```python
# # ```python
# # ```python
# class IteratedPermutationAndCoolingOptimizer:
#     def __init__(self, budget, dim):
#         super().__init__(budget, dim)
        
#     def __call__(self, func):
#         bounds = (-5.0, 5.0)
#         population = super().__call__(func, bounds, budget)
        
#         # Select the fittest individual using iterated permutation with cooling
#         # and return the best point found so far
#         best_point = np.random.uniform(bounds[0], bounds[1], dim)
#         best_value = func(best_point)
#         for _ in range(100):
#             # Select the fittest individual
#             fittest_individual = np.argmax([func(individual) for individual in population])
#             # Generate a new population by iterated permutation
#             new_population = population[:100]
#             new_population = [fittest_individual + np.random.uniform(-1.0, 1.0, dim) for individual in new_population]
#             # Evaluate the new population
#             new_evaluations = [func(individual) for individual in new_population]
#             # Select the fittest new individual
#             new_fittest_individual = np.argmax(new_evaluations)
#             # Update the population with the new individual and bounds
#             population = new_population + [new_fittest_individual]
#             # Update the best point and value
#             best_point = fittest_individual
#             best_value = func(best_point)
#         return best_point, best_value

# # Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# # Code: 
# # ```python
# # ```python
# # ```python
# # ```python
# # Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# # Code: 
# # ```python
# # ```python
# # ```python
# class IteratedPermutationAndCoolingOptimizer:
#     def __init__(self, budget, dim):
#         super().__init__(budget, dim)
        
#     def __call__(self, func):
#         bounds = (-5.0, 5.0)
#         population = super().__call__(func, bounds, budget)
        
#         # Select the fittest individual using iterated permutation with cooling
#         # and return the best point found so far
#         best_point = np.random.uniform(bounds[0], bounds[1], dim)
#         best_value = func(best_point)
#         for _ in range(100):
#             # Select the fittest individual
#             fittest_individual = np.argmax([func(individual) for individual in population])
#             # Generate a new population by iterated permutation
#             new_population = population[:100]
#             new_population = [fittest_individual + np.random.uniform(-1.0, 1.0, dim) for individual in new_population]
#             # Evaluate the new population
#             new_evaluations = [func(individual) for individual in new_population]
#             # Select the fittest new individual
#             new_fittest_individual = np.argmax(new_evaluations)
#             # Update the population with the new individual and bounds
#             population = new_population + [new_fittest_individual]
#             # Update the best point and value
#             best_point = fittest_individual
#             best_value = func(best_point)
#         return best_point, best_value

# # Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# # Code: 
# # ```python
# # ```python
# # ```python
# # ```python
# class IteratedPermutationAndCoolingOptimizer:
#     def __init__(self, budget, dim):
#         super().__init__(budget, dim)
        
#     def __call__(self, func):
#         # Generate a new population by iterated permutation
#         bounds = (-5.0, 5.0)
#         population = [np.random.uniform(bounds[0], bounds[1], dim) for _ in range(100)]
        
#         # Evaluate the new population
#         new_evaluations = [func(individual) for individual in population]
        
#         # Select the fittest new individual
#         new_fittest_individual = np.argmax(new_evaluations)
        
#         # Update the population with the new individual and bounds
#         population = population + [new_fittest_individual]
        
#         # Return the fittest individual in the final population
#         return population[np.argmax([func(individual) for individual in population])]