import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_initial_population()
        self.fitnesses = [self.evaluate_function(func) for func in self.population]
        self.selection_rate = 0.45
        self.crossover_rate = 0.5
        self.mutation_rate = 0.01

    def generate_initial_population(self):
        # Generate a random population of functions with random parameters
        return [self.generate_function() for _ in range(self.budget)]

    def generate_function(self):
        # Generate a random function with random parameters
        return np.random.uniform(-5.0, 5.0, self.dim)

    def evaluate_function(self, func):
        # Evaluate the function using the given parameters
        return func(self.dim)

    def __call__(self, func):
        # Optimize the function using the given parameters
        for _ in range(self.budget):
            # Select the best individual
            selected_func = self.select_best_individual()

            # Generate a new individual by crossover and mutation
            new_func = self.crossover_and_mutation(selected_func)

            # Evaluate the new individual
            fitness = self.evaluate_function(new_func)

            # Update the best individual
            if fitness > self.fitnesses[-1]:
                self.population[-1] = new_func
                self.fitnesses[-1] = fitness

        # Return the best individual
        return self.population[-1]

    def select_best_individual(self):
        # Select the best individual based on the selection rate
        return self.population[np.random.choice(len(self.population), self.selection_rate)]

    def crossover_and_mutation(self, parent_func):
        # Perform crossover and mutation on the parent function
        child_func = parent_func[:self.dim // 2] + parent_func[self.dim // 2 + 1:]
        child_func = self.mutate(child_func)
        return child_func

    def mutate(self, func):
        # Randomly mutate the function
        idx = random.randint(0, len(func) - 1)
        func[idx] += np.random.uniform(-1, 1)
        return func

    def print_best_solution(self):
        # Print the best solution
        print("Best solution:", self.population[0])

# Description: Evolutionary Algorithm for Black Box Optimization
# Code:
# ```python
# ```python
# import numpy as np
# import random

# class EvolutionaryAlgorithm:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population = self.generate_initial_population()
#         self.fitnesses = [self.evaluate_function(func) for func in self.population]
#         self.selection_rate = 0.45
#         self.crossover_rate = 0.5
#         self.mutation_rate = 0.01

#     def generate_initial_population(self):
#         # Generate a random population of functions with random parameters
#         return [self.generate_function() for _ in range(self.budget)]

#     def generate_function(self):
#         # Generate a random function with random parameters
#         return np.random.uniform(-5.0, 5.0, self.dim)

#     def evaluate_function(self, func):
#         # Evaluate the function using the given parameters
#         return func(self.dim)

#     def __call__(self, func):
#         # Optimize the function using the given parameters
#         for _ in range(self.budget):
#             # Select the best individual
#             selected_func = self.select_best_individual()

#             # Generate a new individual by crossover and mutation
#             new_func = self.crossover_and_mutation(selected_func)

#             # Evaluate the new individual
#             fitness = self.evaluate_function(new_func)

#             # Update the best individual
#             if fitness > self.fitnesses[-1]:
#                 self.population[-1] = new_func
#                 self.fitnesses[-1] = fitness

#     def select_best_individual(self):
#         # Select the best individual based on the selection rate
#         return self.population[np.random.choice(len(self.population), self.selection_rate)]

#     def crossover_and_mutation(self, parent_func):
#         # Perform crossover and mutation on the parent function
#         child_func = parent_func[:self.dim // 2] + parent_func[self.dim // 2 + 1:]
#         child_func = self.mutate(child_func)
#         return child_func

#     def mutate(self, func):
#         # Randomly mutate the function
#         idx = random.randint(0, len(func) - 1)
#         func[idx] += np.random.uniform(-1, 1)
#         return func

#     def print_best_solution(self):
#         # Print the best solution
#         print("Best solution:", self.population[0])

# # Description: Evolutionary Algorithm for Black Box Optimization
# # Code:
# # ```python
# # ```
# algorithm = EvolutionaryAlgorithm(100, 10)
# algorithm.print_best_solution()