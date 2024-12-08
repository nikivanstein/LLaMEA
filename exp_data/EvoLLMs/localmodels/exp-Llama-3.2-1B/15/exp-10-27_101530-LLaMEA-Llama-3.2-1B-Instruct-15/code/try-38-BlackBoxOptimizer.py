import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = self.initialize_population()
        self.population_history = []

    def initialize_population(self):
        # Initialize the population with random individuals
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        # Evaluate the function for each individual in the population
        while self.func_evaluations < self.budget:
            # Evaluate the function for each individual in the population
            fitness_values = [func(individual) for individual in self.population]
            # Select the fittest individuals
            fittest_individuals = self.select_fittest_individuals(fitness_values)
            # Create new individuals by mutation
            new_individuals = self.mutate(fittest_individuals)
            # Replace the old population with the new individuals
            self.population = new_individuals
            # Update the population history
            self.population_history.append((self.func_evaluations, self.population, fitness_values))
            # Increment the function evaluations
            self.func_evaluations += 1
        # Return the best individual found so far
        return self.population[0]

    def select_fittest_individuals(self, fitness_values):
        # Select the fittest individuals based on their fitness values
        fittest_individuals = sorted(zip(fitness_values, self.population), key=lambda x: x[0], reverse=True)[:self.population_size // 2]
        return fittest_individuals

    def mutate(self, individuals):
        # Mutate the individuals by changing a random element
        mutated_individuals = []
        for individual in individuals:
            mutated_individual = list(individual)
            for i in range(len(individual)):
                if random.random() < 0.1:
                    mutated_individual[i] += random.uniform(-1, 1)
            mutated_individuals.append(tuple(mutated_individual))
        return mutated_individuals

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = self.initialize_population()
        self.population_history = []
        self.mutant_ratio = 0.1

    def initialize_population(self):
        # Initialize the population with random individuals
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        # Evaluate the function for each individual in the population
        while self.func_evaluations < self.budget:
            # Evaluate the function for each individual in the population
            fitness_values = [func(individual) for individual in self.population]
            # Select the fittest individuals
            fittest_individuals = self.select_fittest_individuals(fitness_values)
            # Create new individuals by mutation
            new_individuals = self.mutate(fittest_individuals)
            # Replace the old population with the new individuals
            self.population = new_individuals
            # Update the population history
            self.population_history.append((self.func_evaluations, self.population, fitness_values))
            # Increment the function evaluations
            self.func_evaluations += 1
        # Return the best individual found so far
        return self.population[0]

    def select_fittest_individuals(self, fitness_values):
        # Select the fittest individuals based on their fitness values
        fittest_individuals = sorted(zip(fitness_values, self.population), key=lambda x: x[0], reverse=True)[:self.population_size // 2]
        return fittest_individuals

    def mutate(self, individuals):
        # Mutate the individuals by changing a random element
        mutated_individuals = []
        for individual in individuals:
            mutated_individual = list(individual)
            for i in range(len(individual)):
                if random.random() < self.mutant_ratio:
                    mutated_individual[i] += random.uniform(-1, 1)
            mutated_individuals.append(tuple(mutated_individual))
        return mutated_individuals

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            point = self.population[self.func_evaluations % len(self.population)]
            func_value = func(point)
            if func_value > 0:
                return point
        return self.search_space[0], self.search_space[1]

# Example usage:
optimizer = NovelMetaheuristicOptimizer(100, 10)
func = lambda x: x**2
best_individual = optimizer(__call__, func)
print("Best individual:", best_individual)
print("Best fitness:", func(best_individual))