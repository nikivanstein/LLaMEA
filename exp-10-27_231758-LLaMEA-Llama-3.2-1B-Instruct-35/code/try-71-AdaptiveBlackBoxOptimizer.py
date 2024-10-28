import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_black_box(self, func, bounds, initial_point, mutation_rate, selection_strategy, num_generations):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Parameters:
        func (function): The black box function to optimize.
        bounds (list): The search space bounds for each dimension.
        initial_point (list): The initial point for the population.
        mutation_rate (float): The mutation rate for the population.
        selection_strategy (str): The selection strategy for the population.
        num_generations (int): The number of generations to evolve.

        Returns:
        dict: A dictionary containing the best solution, its score, and the number of evaluations.
        """
        # Initialize the population with the initial point
        population = [initial_point]
        for _ in range(num_generations):
            # Select the fittest individuals
            fitness = [func(x) for x in population]
            idx = np.argsort(fitness)
            population = [population[i] for i in idx[:int(self.func_evals/2)]] + \
                       [population[idx[int(self.func_evals/2)]]]
            # Perform mutation
            for _ in range(self.func_evals//2):
                idx = np.random.choice(population, 2, replace=False)
                idx = np.sort(idx)
                mutation_rate = np.random.rand() * 0.1
                if mutation_rate < 0.5:
                    population[idx[0]] += np.random.uniform(-1, 1)
                    population[idx[1]] += np.random.uniform(-1, 1)
            # Perform selection
            fitness = [func(x) for x in population]
            idx = np.argsort(fitness)
            population = [population[i] for i in idx]

        # Evaluate the best solution
        best_solution = population[0]
        best_fitness = func(best_solution)
        best_score = best_fitness

        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'best_score': best_score,
            'num_evaluations': self.func_evals
        }

# One-line description: Adaptive Black Box Optimization using Differential Evolution
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer: A metaheuristic algorithm for black box optimization using Differential Evolution.
# ```