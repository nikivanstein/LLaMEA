import random
import numpy as np

class BMBA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

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
        # Select a random mutation point
        mutation_point = random.randint(0, self.dim)
        # Swap the mutation point with a random point in the search space
        mutated_individual = individual[:mutation_point] + (random.uniform(self.search_space[0], self.search_space[1]),) + individual[mutation_point+1:]
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, self.dim)
        # Create a child individual by combining the parents
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

# Description: Novel Metaheuristic Algorithm for Black Box Optimization (BMBA)
# Code: 
# ```python
# import random
# import numpy as np
# import matplotlib.pyplot as plt

# class BMBA:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = (-5.0, 5.0)
#         self.func_evaluations = 0

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
#         # Select a random mutation point
#         mutation_point = random.randint(0, self.dim)
#         # Swap the mutation point with a random point in the search space
#         mutated_individual = individual[:mutation_point] + (random.uniform(self.search_space[0], self.search_space[1]),) + individual[mutation_point+1:]
#         return mutated_individual

#     def crossover(self, parent1, parent2):
#         # Select a random crossover point
#         crossover_point = random.randint(0, self.dim)
#         # Create a child individual by combining the parents
#         child = parent1[:crossover_point] + parent2[crossover_point:]
#         return child

#     def evolve(self, population, mutation_rate, crossover_rate):
#         # Initialize a new population
#         new_population = []
#         for _ in range(10):  # Evolve for 10 generations
#             # Evaluate the fitness of each individual in the population
#             fitnesses = [self.__call__(func) for func in population]
#             # Select parents using tournament selection
#             parents = np.array([fitnesses[i] for i in range(len(fitnesses)) if random.random() < 0.1]).argsort()[:5]
#             # Create offspring using crossover and mutation
#             offspring = [self.crossover(parent1, parent2) for parent1, parent2 in zip(parents, parents[1:])]
#             # Mutate the offspring
#             mutated_offspring = [self.mutate(offspring[i]) for i in range(len(offspring))]
#             # Add the mutated offspring to the new population
#             new_population.extend(mutated_offspring)
#         # Replace the old population with the new population
#         population[:] = new_population
#         # Print the best individual in the new population
#         print(f'Best individual: {np.max(fitnesses)}')
#         return population

# # Example usage:
# optimizer = BMBA(budget=100, dim=10)
# individuals = [optimizer.__call__(func) for func in [lambda x: np.sin(x), lambda x: x**2]]
# optimizer.evolve(individuals, mutation_rate=0.01, crossover_rate=0.7)