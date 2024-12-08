import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.new_individuals = []
        self.experiment_name = "Novel Metaheuristic Algorithm for Black Box Optimization"
        self.score = -np.inf

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

    def mutate(self, individual):
        # Refine the strategy by changing the individual lines
        if random.random() < 0.15:
            # Change the first line of the individual
            individual[0] = random.uniform(self.search_space[0], self.search_space[1])
            # Change the second line of the individual
            individual[1] = random.uniform(self.search_space[0], self.search_space[1])
        return individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual using the given function
        func_value = self.func(individual)
        # Return the fitness value
        return func_value

    def select(self, individuals):
        # Select the best individuals based on their fitness values
        best_individuals = sorted(individuals, key=self.evaluate_fitness, reverse=True)
        # Select the top k individuals
        selected_individuals = [individual for individual in individuals if individual in best_individuals[:self.budget]]
        return selected_individuals

    def update(self, new_individuals, new_fitness):
        # Update the population with the new individuals and their fitness values
        self.new_individuals = new_individuals
        self.score = new_fitness

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 