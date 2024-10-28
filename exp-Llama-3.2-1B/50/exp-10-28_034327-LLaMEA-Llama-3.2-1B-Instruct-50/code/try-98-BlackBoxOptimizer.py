import numpy as np

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

    def novel_black_box_optimization(self, func, budget, dim, cooling_rate=0.99, mutation_rate=0.1):
        """
        Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm.

        Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
        Code: 
        ```python
        # Initialize the population with random points in the search space
        population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

        # Initialize the best point and fitness
        best_point = None
        best_fitness = -np.inf

        # Iterate until the budget is exceeded or the best point is found
        for _ in range(budget):
            # Evaluate the function at each point in the population
            fitness = [func(point) for point in population]
            # Get the index of the point with the highest fitness
            idx = np.argmax(fitness)
            # Get the point with the highest fitness
            best_point = population[idx]
            # Update the best fitness
            best_fitness = np.max(fitness)

            # Refine the strategy by changing the direction of the search
            if np.random.rand() < cooling_rate:
                # Generate a new point by adding a small random value to the current point
                new_point = best_point + np.random.uniform(-1.0, 1.0, self.dim)
                # Evaluate the function at the new point
                new_fitness = func(new_point)
                # Check if the new point is within the bounds
                if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                    # If the new point is within bounds, update the function value
                    self.func_evals += 1
                    population[idx] = new_point
                    # Update the best point and fitness
                    best_point = new_point
                    best_fitness = np.max(fitness)

            # Apply mutation to the current point
            if np.random.rand() < mutation_rate:
                # Generate a random value to mutate
                mutate_value = np.random.uniform(-1.0, 1.0, self.dim)
                # Mutate the current point
                mutated_point = best_point + mutate_value
                # Evaluate the function at the mutated point
                mutated_fitness = func(mutated_point)
                # Check if the mutated point is within the bounds
                if -5.0 <= mutated_point[0] <= 5.0 and -5.0 <= mutated_point[1] <= 5.0:
                    # If the mutated point is within bounds, update the function value
                    self.func_evals += 1
                    population[idx] = mutated_point
                    # Update the best point and fitness
                    best_point = mutated_point
                    best_fitness = np.max(fitness)

        # Return the best point found
        return best_point

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# import numpy as np
# import random
# import time

# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0

#     def __call__(self, func):
#         while self.func_evals < self.budget:
#             # Generate a random point in the search space
#             point = np.random.uniform(-5.0, 5.0, self.dim)
#             # Evaluate the function at the point
#             value = func(point)
#             # Check if the point is within the bounds
#             if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
#                 # If the point is within bounds, update the function value
#                 self.func_evals += 1
#                 return value
#         # If the budget is exceeded, return the best point found so far
#         return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

#     def novel_black_box_optimization(self, func, budget, dim, cooling_rate=0.99, mutation_rate=0.1):
#         """
#         Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm.

#         Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
#         Code: 
#         ```python
#         # Initialize the population with random points in the search space
#         population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

#         # Initialize the best point and fitness
#         best_point = None
#         best_fitness = -np.inf

#         # Iterate until the budget is exceeded or the best point is found
#         for _ in range(budget):
#             # Evaluate the function at each point in the population
#             fitness = [func(point) for point in population]
#             # Get the index of the point with the highest fitness
#             idx = np.argmax(fitness)
#             # Get the point with the highest fitness
#             best_point = population[idx]
#             # Update the best fitness
#             best_fitness = np.max(fitness)

#             # Refine the strategy by changing the direction of the search
#             if np.random.rand() < cooling_rate:
#                 # Generate a new point by adding a small random value to the current point
#                 new_point = best_point + np.random.uniform(-1.0, 1.0, self.dim)
#                 # Evaluate the function at the new point
#                 new_fitness = func(new_point)
#                 # Check if the new point is within the bounds
#                 if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
#                     # If the new point is within bounds, update the function value
#                     self.func_evals += 1
#                     population[idx] = new_point
#                     # Update the best point and fitness
#                     best_point = new_point
#                     best_fitness = np.max(fitness)

#             # Apply mutation to the current point
#             if np.random.rand() < mutation_rate:
#                 # Generate a random value to mutate
#                 mutate_value = np.random.uniform(-1.0, 1.0, self.dim)
#                 # Mutate the current point
#                 mutated_point = best_point + mutate_value
#                 # Evaluate the function at the mutated point
#                 mutated_fitness = func(mutated_point)
#                 # Check if the mutated point is within the bounds
#                 if -5.0 <= mutated_point[0] <= 5.0 and -5.0 <= mutated_point[1] <= 5.0:
#                     # If the mutated point is within bounds, update the function value
#                     self.func_evals += 1
#                     population[idx] = mutated_point
#                     # Update the best point and fitness
#                     best_point = mutated_point
#                     best_fitness = np.max(fitness)

#         # Return the best point found
#         return best_point

# # Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# # Code: 
# # ```python
# # import numpy as np
# # import random
# # import time

# def novel_bbo_optimization(func, budget, dim, cooling_rate=0.99, mutation_rate=0.1):
#     """
#     Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm.

#     Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
#     Code: 
#     ```python
#     # Initialize the population with random points in the search space
#     population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

#     # Initialize the best point and fitness
#     best_point = None
#     best_fitness = -np.inf

#     # Iterate until the budget is exceeded or the best point is found
#     for _ in range(budget):
#         # Evaluate the function at each point in the population
#         fitness = [func(point) for point in population]
#         # Get the index of the point with the highest fitness
#         idx = np.argmax(fitness)
#         # Get the point with the highest fitness
#         best_point = population[idx]
#         # Update the best fitness
#         best_fitness = np.max(fitness)

#         # Refine the strategy by changing the direction of the search
#         if np.random.rand() < cooling_rate:
#             # Generate a new point by adding a small random value to the current point
#             new_point = best_point + np.random.uniform(-1.0, 1.0, dim)
#             # Evaluate the function at the new point
#             new_fitness = func(new_point)
#             # Check if the new point is within the bounds
#             if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
#                 # If the new point is within bounds, update the function value
#                 self.func_evals += 1
#                 population[idx] = new_point
#                 # Update the best point and fitness
#                 best_point = new_point
#                 best_fitness = np.max(fitness)

#             # Apply mutation to the current point
#             if np.random.rand() < mutation_rate:
#                 # Generate a random value to mutate
#                 mutate_value = np.random.uniform(-1.0, 1.0, dim)
#                 # Mutate the current point
#                 mutated_point = best_point + mutate_value
#                 # Evaluate the function at the mutated point
#                 mutated_fitness = func(mutated_point)
#                 # Check if the mutated point is within the bounds
#                 if -5.0 <= mutated_point[0] <= 5.0 and -5.0 <= mutated_point[1] <= 5.0:
#                     # If the mutated point is within bounds, update the function value
#                     self.func_evals += 1
#                     population[idx] = mutated_point
#                     # Update the best point and fitness
#                     best_point = mutated_point
#                     best_fitness = np.max(fitness)

#         # Return the best point found
#         return best_point

# # Example usage
# optimizer = BlackBoxOptimizer(budget=100, dim=10)
# func = lambda point: point[0]**2 + point[1]**2
# best_point = optimizer.novel_bbo_optimization(func, 100, 10)
# print(best_point)