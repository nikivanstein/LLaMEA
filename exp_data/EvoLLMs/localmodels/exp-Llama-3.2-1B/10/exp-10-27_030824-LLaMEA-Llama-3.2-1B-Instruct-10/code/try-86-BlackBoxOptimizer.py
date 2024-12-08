import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Select a random mutation point
        mutation_point = random.randint(0, self.dim - 1)
        # Swap the element at the mutation point with a random element from the search space
        individual[mutation_point], individual[mutation_point + random.randint(0, self.dim - 1)] = individual[mutation_point + random.randint(0, self.dim - 1)], individual[mutation_point]
        # Ensure the mutation point is within the search space
        individual[mutation_point] = np.clip(individual[mutation_point], self.search_space[0], self.search_space[1])
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, self.dim - 1)
        # Create a child individual by combining the parents
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        # Return the child individual
        return child

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# class NovelMetaheuristicOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = [-5.0, 5.0]
#         self.func_evaluations = 0

#     def __call__(self, func):
#         # Ensure the function evaluations do not exceed the budget
#         if self.func_evaluations < self.budget:
#             # Generate a random point in the search space
#             point = np.random.uniform(self.search_space[0], self.search_space[1])
#             # Evaluate the function at the point
#             evaluation = func(point)
#             # Increment the function evaluations
#             self.func_evaluations += 1
#             # Return the point and its evaluation
#             return point, evaluation
#         else:
#             # If the budget is reached, return a default point and evaluation
#             return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

#     def mutate(self, individual):
#         # Select a random mutation point
#         mutation_point = random.randint(0, self.dim - 1)
#         # Swap the element at the mutation point with a random element from the search space
#         individual[mutation_point], individual[mutation_point + random.randint(0, self.dim - 1)] = individual[mutation_point + random.randint(0, self.dim - 1)], individual[mutation_point]
#         # Ensure the mutation point is within the search space
#         individual[mutation_point] = np.clip(individual[mutation_point], self.search_space[0], self.search_space[1])
#         return individual

#     def crossover(self, parent1, parent2):
#         # Select a random crossover point
#         crossover_point = random.randint(0, self.dim - 1)
#         # Create a child individual by combining the parents
#         child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
#         # Return the child individual
#         return child

# optimizer = NovelMetaheuristicOptimizer(100, 10)
# optimizer.func_evaluations = 0
# for _ in range(100):
#     print(optimizer())