# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        def evaluate_fitness(individual):
            return self.func(individual)

        def mutate(individual):
            if np.random.rand() < 0.05:
                return individual + np.random.uniform(-5.0, 5.0, size=self.dim)
            else:
                return individual

        population_size = self.budget // self.dim
        population = [evaluate_fitness(random.choice(self.search_space)) for _ in range(population_size)]
        while True:
            next_generation = []
            for _ in range(self.budget - self.dim):
                parent1, parent2 = random.sample(population, 2)
                child = mutate(parent1 + parent2)
                next_generation.append(child)
            population = next_generation
            population = np.vstack((population, [evaluate_fitness(individual) for individual in population]))

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        def evaluate_fitness(individual):
            return self.func(individual)

        def mutate(individual):
            if np.random.rand() < 0.05:
                return individual + np.random.uniform(-5.0, 5.0, size=self.dim)
            else:
                return individual

        population_size = self.budget // self.dim
        population = [evaluate_fitness(random.choice(self.search_space)) for _ in range(population_size)]
        while True:
            next_generation = []
            for _ in range(self.budget - self.dim):
                parent1, parent2 = random.sample(population, 2)
                child = mutate(parent1 + parent2)
                next_generation.append(child)
            population = next_generation
            population = np.vstack((population, [evaluate_fitness(individual) for individual in population]))

            # Refine the strategy
            if np.random.rand() < 0.05:
                mutation_rate = random.uniform(0.01, 0.1)
                for individual in population:
                    if np.random.rand() < mutation_rate:
                        parent1, parent2 = random.sample(population, 2)
                        child = mutate(parent1 + parent2)
                        population = [evaluate_fitness(individual) for individual in population]
            else:
                population = np.delete(population, np.random.randint(0, self.budget // self.dim, size=self.dim), axis=0)

# Example usage:
optimizer = BBOBOptimizer(100, 10)
optimizer.__call__(lambda x: np.sum(x))