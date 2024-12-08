import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Evaluate the black box function using the budget
        evaluations = [func(solution) for solution in self.population]
        # Select the best solution based on the budget
        selected_solution = self.select_best_solution(evaluations, self.budget)
        # Optimize the selected solution
        optimized_solution = self.optimize_solution(selected_solution)
        # Return the score of the optimized solution
        return self.score_function(optimized_solution)

    def select_best_solution(self, evaluations, budget):
        # Select the solution that has the highest score based on the budget
        # Use the probability of 0.45 to refine the strategy
        probabilities = [evaluations[i] / budget for i in range(self.population_size)]
        selected_index = np.random.choice(self.population_size, p=probabilities)
        return self.population[selected_index]

    def optimize_solution(self, solution):
        # Use a novel metaheuristic algorithm to optimize the solution
        # Use the probability of 0.45 to refine the strategy
        probabilities = [0.55 if random.random() < 0.45 else 0.45 for _ in range(self.dim)]
        selected_index = np.random.choice(self.dim, p=probabilities)
        return self.population[selected_index]

    def score_function(self, solution):
        # Evaluate the black box function using the score function
        # Use the probability of 0.45 to refine the strategy
        probabilities = [0.55 if random.random() < 0.45 else 0.45 for _ in range(self.dim)]
        selected_index = np.random.choice(self.dim, p=probabilities)
        return self.population[selected_index]