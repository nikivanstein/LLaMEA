import numpy as np
import random

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.memory = []
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.pbest_count = np.zeros((self.population_size, self.dim))
        self.random_solution = np.random.uniform(-5.0, 5.0, size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a new population
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Evaluate the population
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the global best solution
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
            # Update the memory with the best solutions
            self.memory.append(self.pbest[self.pbest_count == i, :])
            # Apply memetic operators
            for i in range(self.population_size):
                # Select a random solution from the memory
                random_solution = np.random.choice(self.memory, size=1)[0]
                # Mutate the solution
                mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
                mutated_solution = random_solution + mutation
                # Apply crossover
                crossover = np.random.choice([0, 1], size=self.dim)
                if crossover[0] == 1:
                    mutated_solution[crossover[1]] = random_solution[crossover[1]]
                # Replace the solution with the mutated solution
                population[i, :] = mutated_solution
            # Evaluate the population again
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the global best solution
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
        return self.gbest

    def mutate(self, solution):
        # Only change 2.0% of the code
        # Select 10% of the dimensions randomly
        indices_to_mutate = np.random.choice(self.dim, size=np.random.randint(1, int(self.dim * 0.02)), replace=False)
        # Mutate the selected dimensions
        mutated_solution = solution.copy()
        for index in indices_to_mutate:
            mutated_solution[index] = np.random.uniform(-0.1, 0.1)
        return mutated_solution

# Example usage:
# ```python
# from llamea import HybridMetaheuristic
# from bbo_benchmarks import BBOB
# 
# # Define the objective function
# def func(x):
#     return np.sum(x**2)
# 
# # Create an instance of the HybridMetaheuristic class
# hybrid = HybridMetaheuristic(budget=100, dim=5)
# 
# # Evaluate the objective function on the population
# population = np.random.uniform(-5.0, 5.0, size=(hybrid.population_size, hybrid.dim))
# evaluations = func(population)
# 
# # Update the population with the best solutions
# for i in range(hybrid.population_size):
#     if evaluations[i] < evaluations[hybrid.pbest_count[i, :]]:
#         hybrid.pbest[i, :] = population[i, :]
#         hybrid.pbest_count[i, :] = i
# 
# # Update the global best solution
# min_evaluation = np.min(evaluations)
# if min_evaluation < hybrid.gbest[np.argmin(hybrid.gbest)]:
#     hybrid.gbest = evaluations[np.argmin(hybrid.gbest)]
# 
# print(hybrid.gbest)