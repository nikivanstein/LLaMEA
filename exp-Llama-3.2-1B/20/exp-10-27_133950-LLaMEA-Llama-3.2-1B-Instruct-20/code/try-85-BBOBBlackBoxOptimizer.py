# Description: Novel metaheuristic algorithm to optimize black box functions using a dynamic search space.
# Code: 
# ```python
import numpy as np
import random
import copy

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def evolve(self, individual, population_size):
        # Select parents using tournament selection
        parents = []
        for _ in range(population_size):
            tournament_size = random.randint(2, self.dim)
            tournament = random.sample([individual] * tournament_size, tournament_size)
            winner = max(tournament, key=lambda x: self.evaluate_fitness(x))
            parents.append(winner)

        # Create offspring using crossover
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child = (parent1 + parent2) / 2
            offspring.append(child)

        # Select fittest individual
        fittest_individual = max(offspring, key=self.evaluate_fitness)

        return fittest_individual

    def evaluate_fitness(self, individual):
        return copy.deepcopy(self.func_evaluations)

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Novel heuristic algorithm: Evolutionary Algorithm with Dynamic Search Space (EAS-DSS)
# Code: 
# ```python
# EAS-DSS: Evolutionary Algorithm with Dynamic Search Space
# Description: Novel metaheuristic algorithm to optimize black box functions using a dynamic search space.
# Code: 
# ```python
# ```python
# ```python
# import numpy as np
# import random
# import copy

class EASDSSBBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def evolve(self, individual, population_size):
        # Select parents using tournament selection
        parents = []
        for _ in range(population_size):
            tournament_size = random.randint(2, self.dim)
            tournament = random.sample([individual] * tournament_size, tournament_size)
            winner = max(tournament, key=lambda x: self.evaluate_fitness(x))
            parents.append(winner)

        # Create offspring using crossover
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child = (parent1 + parent2) / 2
            offspring.append(child)

        # Select fittest individual
        fittest_individual = max(offspring, key=self.evaluate_fitness)

        return fittest_individual

    def evaluate_fitness(self, individual):
        return copy.deepcopy(self.func_evaluations)

# Example usage:
optimizer = EASDSSBBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)