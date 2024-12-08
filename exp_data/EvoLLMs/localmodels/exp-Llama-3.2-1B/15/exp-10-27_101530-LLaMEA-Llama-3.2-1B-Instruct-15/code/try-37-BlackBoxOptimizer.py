import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        def mutation(individual):
            if random.random() < 0.15:
                return individual[:random.randint(0, len(individual) - 1)] + [random.uniform(-5.0, 5.0)]
            return individual

        def selection(individual):
            if random.random() < 0.15:
                return random.choice(self.search_space)
            return individual

        def crossover(parent1, parent2):
            if random.random() < 0.15:
                child = parent1[:random.randint(0, len(parent1) - 1)] + [random.uniform(-5.0, 5.0)]
                child[-1] = parent2[-1]
                return child
            return parent1 + parent2

        def selection_sort(individual):
            for i in range(len(individual)):
                min_idx = i
                for j in range(i + 1, len(individual)):
                    if random.random() < 0.15:
                        min_idx = j
                individual[i], individual[min_idx] = individual[min_idx], individual[i]

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

class BlackBoxOptimizerEvolutionary(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [BlackBoxOptimizerEvolutionary.Bird(self.budget, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        population = self.population
        for _ in range(100):  # Evolve for 100 generations
            for individual in population:
                if random.random() < 0.15:
                    individual = mutation(individual)
                if random.random() < 0.15:
                    individual = selection(individual)
                if random.random() < 0.15:
                    individual = crossover(individual, individual)
                individual = selection_sort(individual)
            population = [individual for individual in population if random.random() < 0.15]
        # Return the best individual
        return self.population[0]