import numpy as np
from scipy.optimize import minimize
import random
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.population = deque(maxlen=1000)  # Population of candidate individuals
        self.logger = random.getrandbits(32)  # Random logger for mutation

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            self.population.append(best_x)  # Add best individual to population
            if len(self.population) > self.budget:
                self.population.popleft()  # Remove oldest individual
            self.logger ^= random.getrandbits(32)  # Update random logger
            if random.random() < 0.45:  # Refine strategy with probability 0.45
                for _ in range(1, self.dim):
                    new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                    new_value = self.func(new_x)
                    if new_value < best_value:
                        best_x = new_x
                        best_value = new_value
                self.population.append(best_x)  # Add refined individual to population
        return self.population.popleft()  # Return best individual

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 