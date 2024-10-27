import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.iterations = 0

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
        # Define the mutation function
        def mutate_point(point):
            # Generate a random mutation factor
            mutation_factor = random.uniform(0.1, 0.3)
            # Apply the mutation
            mutated_point = (point[0] + mutation_factor * (point[1] - point[0]), point[1] + mutation_factor * (point[0] - point[1]))
            return mutated_point

        # Apply the mutation to the individual
        mutated_individual = self.evaluate_fitness(mutate_point(individual))
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Define the crossover function
        def crossover_point(point1, point2):
            # Generate a random crossover point
            crossover_point = random.uniform(min(point1[0], point2[0]), max(point1[0], point2[0]))
            # Split the points
            child1 = (point1[0] if point1[0] <= point2[0] else point2[0], point1[1])
            child2 = (point2[0] if point1[0] <= point2[0] else point1[0], point2[1])
            # Return the children
            return child1, child2

        # Apply the crossover
        child1, child2 = crossover_point(parent1, parent2)
        return child1, child2

    def __str__(self):
        return "Novel Metaheuristic Algorithm for Black Box Optimization"

    def evaluate_fitness(self, individual):
        # Define the fitness function
        def fitness(individual):
            # Evaluate the function at the individual
            func_value = individual[0]**2 + individual[1]**2
            return func_value

        # Evaluate the fitness
        fitness_value = fitness(individual)
        return fitness_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np
# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = (-5.0, 5.0)
#         self.func_evaluations = 0
#         self.iterations = 0

#     def __call__(self, func):
#         while self.func_evaluations < self.budget:
#             # Generate a random point in the search space
#             point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
#             # Evaluate the function at the point
#             func_value = func(point)
#             # Increment the function evaluations
#             self.func_evaluations += 1
#             # Check if the point is within the budget
#             if self.func_evaluations < self.budget:
#                 # If not, return the point
#                 return point
#         # If the budget is reached, return the best point found so far
#         return self.search_space[0], self.search_space[1]

#     def mutate(self, individual):
#         # Define the mutation function
#         def mutate_point(point):
#             # Generate a random mutation factor
#             mutation_factor = random.uniform(0.1, 0.3)
#             # Apply the mutation
#             mutated_point = (point[0] + mutation_factor * (point[1] - point[0]), point[1] + mutation_factor * (point[0] - point[1]))
#             return mutated_point

#         # Apply the mutation to the individual
#         mutated_individual = self.evaluate_fitness(mutate_point(individual))
#         return mutated_individual

#     def crossover(self, parent1, parent2):
#         # Define the crossover function
#         def crossover_point(point1, point2):
#             # Generate a random crossover point
#             crossover_point = random.uniform(min(point1[0], point2[0]), max(point1[0], point2[0]))
#             # Split the points
#             child1 = (point1[0] if point1[0] <= point2[0] else point2[0], point1[1])
#             child2 = (point2[0] if point1[0] <= point2[0] else point1[0], point2[1])
#             # Return the children
#             return child1, child2

#         # Apply the crossover
#         child1, child2 = crossover_point(parent1, parent2)
#         return child1, child2

#     def __str__(self):
#         return "Novel Metaheuristic Algorithm for Black Box Optimization"

# test the optimizer
optimizer = BlackBoxOptimizer(100, 10)
best_individual = optimizer.__call__(np.sin)
print(optimizer)
print(best_individual)