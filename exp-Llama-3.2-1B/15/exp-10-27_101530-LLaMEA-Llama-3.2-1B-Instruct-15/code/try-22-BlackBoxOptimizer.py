import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.strategies = {
            'uniform': lambda x: x,
            'bounded': lambda x: self.bounded(x)
        }
        self.current_strategy = 'uniform'

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
            # Update the strategy based on the current fitness
            if self.func_evaluations / self.budget > 0.15:
                self.current_strategy = random.choice(list(self.strategies.keys()))
                self.strategies[self.current_strategy] = self.strategies[self.current_strategy].func
            # If the current strategy is exhausted, switch to a new one
            if self.current_strategy == 'bounded':
                self.current_strategy = random.choice(list(self.strategies.keys()))
                if self.current_strategy == 'bounded':
                    self.strategies[self.current_strategy] = self.bounded
            # If the current strategy is 'bounded', check if the point is within the budget
            if self.current_strategy == 'bounded':
                if self.strategies[self.current_strategy](point) < self.search_space[0]:
                    point = (self.strategies[self.current_strategy](point), self.strategies[self.current_strategy](point))
                elif self.strategies[self.current_strategy](point) > self.search_space[1]:
                    point = (self.strategies[self.current_strategy](point), self.strategies[self.current_strategy](point))
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

class BoundedBlackBoxOptimizer(BlackBoxOptimizer):
    def bounded(self, point):
        return min(max(point, self.search_space[0]), self.search_space[1])

class GeneticBlackBoxOptimizer(BlackBoxOptimizer):
    def __init__(self, budget, dim, mutation_rate, crossover_rate):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def __call__(self, func):
        population = [func(np.random.rand(self.dim)) for _ in range(100)]
        for _ in range(10):
            while True:
                parent1, parent2 = random.sample(population, 2)
                child = self.mutate(parent1, parent2)
                if self.budget < 1000 and self.func_evaluations / self.budget > 0.15:
                    self.func_evaluations += 1
                    child = self.bounded(child)
                if self.func_evaluations / self.budget > 0.15:
                    self.func_evaluations += 1
                    child = self.bounded(child)
                if self.func_evaluations / self.budget > 0.15 and random.random() < self.mutation_rate:
                    self.func_evaluations += 1
                    child = self.mutate(child)
                population.append(child)
        return population[0]

def mutate(individual, parent1, parent2):
    if random.random() < self.mutation_rate:
        return self.mutate_individual(individual, parent1, parent2)
    return parent1

def mutate_individual(individual, parent1, parent2):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def mutate_bounded(individual, parent1, parent2):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual