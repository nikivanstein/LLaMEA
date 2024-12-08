import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def optimize(self, func):
        # Initialize the population
        population = [self.__call__(func) for _ in range(self.population_size)]
        
        # Evaluate the initial population
        initial_evaluations = np.array([func(x) for x in population])
        
        # Initialize the best solution
        best_individual = np.argmax(initial_evaluations)
        best_fitness = initial_evaluations[best_individual]
        
        # Run the evolution
        for _ in range(self.budget):
            # Evaluate the new population
            new_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(population))])
            
            # Select the next generation
            next_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(population, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child = random.uniform(self.search_space[0], self.search_space[1])
                next_population.append(child)
            
            # Replace the old population with the new one
            population = next_population
            
            # Evaluate the new population
            new_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(next_population))])
            
            # Update the best solution
            best_individual = np.argmax(new_evaluations)
            best_fitness = new_evaluations[best_individual]
        
        # Return the best solution
        return population[best_individual], best_fitness

# One-line description with the main idea
# BlackBoxOptimizer: A metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
# import random
# import numpy as np
# from scipy.optimize import minimize

# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.search_space = (-5.0, 5.0)
#         self.population_size = 100
#         self.crossover_rate = 0.5
#         self.mutation_rate = 0.1

#     def __call__(self, func):
#         # Evaluate the function with the given budget
#         func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
#         # Select the top-performing individuals
#         top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
#         # Create a new population by crossover and mutation
#         new_population = []
#         for _ in range(self.population_size):
#             parent1, parent2 = random.sample(top_individuals, 2)
#             child = (parent1 + parent2) / 2
#             if random.random() < self.mutation_rate:
#                 child = random.uniform(self.search_space[0], self.search_space[1])
#             new_population.append(child)
        
#         # Replace the old population with the new one
#         self.population = new_population
        
#         # Evaluate the new population
#         new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
#         # Return the best individual
#         best_individual = np.argmax(new_func_evaluations)
#         return new_population[best_individual]

#     def optimize(self, func):
#         # Initialize the population
#         population = [self.__call__(func) for _ in range(self.population_size)]
        
#         # Evaluate the initial population
#         initial_evaluations = np.array([func(x) for x in population])
        
#         # Initialize the best solution
#         best_individual = np.argmax(initial_evaluations)
#         best_fitness = initial_evaluations[best_individual]
        
#         # Run the evolution
#         for _ in range(self.budget):
#             # Evaluate the new population
#             new_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(population))])
            
#             # Select the next generation
#             next_population = []
#             for _ in range(self.population_size):
#                 parent1, parent2 = random.sample(population, 2)
#                 child = (parent1 + parent2) / 2
#                 if random.random() < self.mutation_rate:
#                     child = random.uniform(self.search_space[0], self.search_space[1])
#                 next_population.append(child)
            
#             # Replace the old population with the new one
#             population = next_population
            
#             # Evaluate the new population
#             new_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(next_population))])
            
#             # Update the best solution
#             best_individual = np.argmax(new_evaluations)
#             best_fitness = new_evaluations[best_individual]
        
#         # Return the best solution
#         return population[best_individual], best_fitness

# Example usage
if __name__ == "__main__":
    optimizer = BlackBoxOptimizer(budget=100, dim=10)
    func = lambda x: x**2
    best_individual, best_fitness = optimizer.optimize(func)
    print("Best individual:", best_individual)
    print("Best fitness:", best_fitness)