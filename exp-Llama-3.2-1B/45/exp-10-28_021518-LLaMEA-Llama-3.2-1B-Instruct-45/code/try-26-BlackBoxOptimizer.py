import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            for _ in range(dim):
                population.append(np.random.uniform(-5.0, 5.0))
        return population

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        def fitness_func(x):
            return evaluate_func(x)

        while len(self.elite) < self.elite_size:
            # Selection
            fitness_values = [fitness_func(x) for x in self.population]
            indices = np.argsort(fitness_values)[:self.population_size]
            self.elite = [self.population[i] for i in indices]

            # Adaptive Mutation Rate
            mutation_rate = 0.1 * self.budget / len(self.elite)
            for individual in self.elite:
                if random.random() < mutation_rate:
                    index = random.randint(0, self.dim - 1)
                    individual[index] += random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = self.elite[:]

        return self.elite[0]

# Evolutionary Algorithm for Black Box Optimization
# Code: 
# ```python
# Genetic Algorithm for Black Box Optimization
# Code: 
# ```
# ```python
# BlackBoxOptimizer: Genetic Algorithm for Black Box Optimization
# Code: 
# ```python
# ```python
# def __call__(self, func):
#     # Adaptive Mutation Rate
#     mutation_rate = 0.1 * self.budget / len(self.elite)
#     for individual in self.elite:
#         if random.random() < mutation_rate:
#             index = random.randint(0, self.dim - 1)
#             individual[index] += random.uniform(-1.0, 1.0)

#     # Selection
#     fitness_values = [fitness_func(x) for x in self.population]
#     indices = np.argsort(fitness_values)[:self.population_size]
#     self.elite = [self.population[i] for i in indices]

#     # Crossover
#     children = []
#     for _ in range(self.population_size // 2):
#         parent1, parent2 = random.sample(self.elite, 2)
#         child = (parent1 + parent2) / 2
#         children.append(child)

#     # Mutation
#     for child in children:
#         if random.random() < 0.1:
#             index = random.randint(0, self.dim - 1)
#             child[index] += random.uniform(-1.0, 1.0)

#     # Replace the elite with the children
#     self.elite = children

#     # Evaluate the fitness of the new population
#     new_population = []
#     for individual in self.elite:
#         fitness_values = [fitness_func(x) for x in individual]
#         new_individual = [x for x, y in zip(individual, fitness_values) if y > 0]
#         new_population.append(new_individual)
#     new_population = np.array(new_population)
#     return new_population