import random
import numpy as np

class BlackBoxOptimizer:
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

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.population = self.generate_initial_population(dim)

    def generate_initial_population(self, dim):
        population = []
        for _ in range(100):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Select the fittest individual
            fittest_individual = self.population[np.argmin([self.evaluate_fitness(individual) for individual in self.population])]
            # Generate a new individual using the fittest individual
            new_individual = self.generate_new_individual(fittest_individual)
            # Evaluate the new individual
            new_fitness = self.evaluate_fitness(new_individual)
            # Check if the new individual is within the budget
            if new_fitness <= self.budget:
                # If so, return the new individual
                return new_individual, new_fitness
            # If not, replace the fittest individual with the new individual
            self.population[np.argmin([self.evaluate_fitness(individual) for individual in self.population])] = new_individual
            self.func_evaluations += 1
        # If the budget is reached, return the best individual found so far
        return self.best_individual, self.best_fitness

    def generate_new_individual(self, fittest_individual):
        # Define the mutation rules
        mutation_rules = {
            'crossover': self.crossover,
           'mutation': self.mutation
        }
        # Generate a new individual by crossover
        if random.random() < 0.5:
            new_individual = fittest_individual[:self.dim//2] + [self.mutation(fittest_individual[self.dim//2], self.mutation(fittest_individual[self.dim//2+1]))]
        else:
            new_individual = fittest_individual[:self.dim] + [self.mutation(fittest_individual[self.dim-1], self.mutation(fittest_individual[self.dim]))]
        return new_individual

    def crossover(self, parent1, parent2):
        # Define the crossover rules
        crossover_rules = {
            'uniform': lambda x, y: random.uniform(x, y),
            'bitwise': lambda x, y: random.randint(x, y)
        }
        # Perform crossover
        child = [crossover_rules['uniform'](x, y) for x, y in zip(parent1, parent2)]
        return child

    def mutation(self, individual, mutation_rule):
        # Define the mutation rules
        mutation_rules = {
            'uniform': lambda x, y: random.uniform(x, y),
            'bitwise': lambda x, y: random.randint(x, y)
        }
        # Perform mutation
        for i in range(len(individual)):
            if random.random() < 0.5:
                individual[i] = mutation_rules['uniform'](individual[i], individual[i+1])
        return individual

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 