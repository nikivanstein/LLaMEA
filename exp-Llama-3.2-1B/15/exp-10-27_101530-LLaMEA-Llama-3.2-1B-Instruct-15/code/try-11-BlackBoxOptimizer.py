import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.new_individual = None
        self.new_point = None
        self.new_fitness = None
        self.population = []

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            if self.population:
                self.new_individual = self.population[0]
                self.new_point = self.new_individual.copy()
                self.new_fitness = func(self.new_point)
            else:
                # If the population is empty, generate a new individual
                self.new_individual = self.evaluate_fitness(self.evaluate_fitness(np.array([random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))]))
                self.new_point = self.new_individual.copy()
                self.new_fitness = func(self.new_point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return self.new_point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        if random.random() < 0.1:
            # Randomly mutate the individual
            mutated_individual = individual.copy()
            mutated_individual[random.randint(0, self.dim-1)] = random.uniform(self.search_space[0], self.search_space[1])
            return mutated_individual
        else:
            # Do not mutate the individual
            return individual

    def evaluate_fitness(self, individual):
        func_value = func(individual)
        return func_value

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
#         self.new_individual = None
#         self.new_point = None
#         self.new_fitness = None
#         self.population = []

#     def __call__(self, func):
#         while self.func_evaluations < self.budget:
#             # Generate a random point in the search space
#             if self.population:
#                 self.new_individual = self.population[0]
#                 self.new_point = self.new_individual.copy()
#                 self.new_fitness = func(self.new_point)
#             else:
#                 # If the population is empty, generate a new individual
#                 self.new_individual = self.evaluate_fitness(np.array([random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))]))
#                 self.new_point = self.new_individual.copy()
#                 self.new_fitness = func(self.new_point)
#             # Increment the function evaluations
#             self.func_evaluations += 1
#             # Check if the point is within the budget
#             if self.func_evaluations < self.budget:
#                 # If not, return the point
#                 return self.new_point
#         # If the budget is reached, return the best point found so far
#         return self.search_space[0], self.search_space[1]

def func(x):
    return np.sin(x)

# Initialize the optimizer
optimizer = BlackBoxOptimizer(100, 10)

# Run the optimizer
optimizer()

# Print the results
print(optimizer.new_point)
print(optimizer.new_fitness)