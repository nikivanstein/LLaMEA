import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

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
            initial_guess = best_x
        return best_x, best_value

    def novel_metaheuristic(self, func, initial_guess, iterations, budget):
        # Novel metaheuristic algorithm for black box optimization using a novel search strategy
        # 
        # The strategy is to change the individual lines of the selected solution to refine its strategy
        # 
        # The probability of changing an individual is 0.45, and the probability of not changing an individual is 0.55
        # 
        # The algorithm starts with the initial guess, and then iteratively applies the search strategy
        # 
        # The algorithm stops when the budget is reached or the best solution is found
        # 

        # Initialize the population with the initial guess
        population = [initial_guess]
        for _ in range(iterations):
            # Generate a new population by changing each individual in the current population
            new_population = []
            for _ in range(self.dim):
                new_individual = [x + random.uniform(-0.01, 0.01) for x in population[-1]]
                new_individual = [x if np.random.rand() < 0.45 else x + random.uniform(-0.01, 0.01) for x in new_individual]
                new_population.append(new_individual)
            # Add the new population to the current population
            population.extend(new_population)

        # Evaluate the best individual in the new population
        best_individual, best_value = self.func(population[0])
        # Evaluate the best individual in the current population
        best_individual, best_value = self.func(initial_guess)

        # Select the best individual based on the probability
        selected_individual = random.choices(population, weights=[best_value / self.budget for _ in range(self.budget)], k=1)[0]

        return selected_individual, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# The algorithm starts with the initial guess, and then iteratively applies the search strategy
# 
# The probability of changing an individual is 0.45, and the probability of not changing an individual is 0.55
# 
# The algorithm stops when the budget is reached or the best solution is found