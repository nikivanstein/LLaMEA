import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        # Select a random parent from the current population
        parent1, parent2 = np.random.choice(self.population, 2, replace=False)
        
        # Create a new individual by combining the parents
        child = np.concatenate((parent1, parent2))
        
        # Apply mutation rules
        if np.random.rand() < 0.05:  # 5% chance of mutation
            # Swap two random genes
            i = np.random.randint(0, self.dim)
            j = np.random.randint(0, self.dim)
            child[i], child[j] = child[j], child[i]
        
        return child

    def evaluate_fitness(self, individual):
        func_value = self.func(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def select_parents(self, population_size):
        # Select parents using tournament selection
        tournament_size = 2
        tournament_results = []
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(self.population, 2, replace=False)
            tournament_results.append(self.evaluate_fitness(parent1))
            tournament_results.append(self.evaluate_fitness(parent2))
        tournament_results = np.array(tournament_results) / 2
        return tournament_results[np.argsort(tournament_results)]

    def update_population(self, population_size):
        # Update population using elitism
        elite_size = 1
        elite_population = self.population[:elite_size]
        new_population = []
        for _ in range(population_size - elite_size):
            parent1, parent2 = self.select_parents(population_size)
            new_individual = self.mutate(parent1)
            new_population.append(new_individual)
        self.population = np.concatenate((elite_population, new_population))

# Description: An evolutionary algorithm for black box optimization
# Code: 
# ```python
# HEBBO: An evolutionary algorithm for black box optimization
# 
# It uses a combination of mutation and selection to refine the solution
# and adapts to the changing fitness landscape
# 
# The algorithm is inspired by the concept of evolution and
# natural selection, where the fittest individuals are selected
# to reproduce and create the next generation
# 
# The probability of mutation is set to 5% to introduce randomness
# and the probability of selection is set to 95% to emphasize
# the importance of the current generation
# 
# The algorithm is designed to handle a wide range of tasks and
# can be used to solve black box optimization problems
# 
# The current population is evaluated using the BBOB test suite
# and the algorithm is updated using elitism to ensure that
# the fittest individuals are selected for reproduction
# 
# The algorithm can be further improved by incorporating
# more advanced techniques such as genetic programming or
# evolutionary strategies