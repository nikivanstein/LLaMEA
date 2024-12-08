import numpy as np
import random

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))

    def __call__(self, func):
        # Adaptive sampling strategy
        adaptive_func = self.adaptive_sampling(func)
        for _ in range(self.budget):
            # Evaluate the function with the current population
            best_func = func(self.population)
            # Update the population with the best function
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Update the adaptive function with the current population
            if np.all(best_func == func(self.population)):
                break
        return func(self.population)

    def adaptive_sampling(self, func):
        # Adaptive sampling strategy
        num_evaluations = 0
        while num_evaluations < self.budget:
            # Evaluate the function with the current population
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(func_evals)
            # Update the adaptive function with the current population
            if np.any(best_func!= func(func_evals)):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Update the adaptive function with the best function
            if np.all(best_func == func(func_evals)):
                break
            num_evaluations += 1
        return func_evals

    def select_strategy(self):
        # Select the adaptive sampling strategy based on the number of evaluations
        if self.num_evaluations < 50:
            return self.adaptive_sampling
        else:
            return self.baseline_sampling

    def baseline_sampling(self):
        # Baseline sampling strategy
        adaptive_func = self.select_strategy()
        for _ in range(self.budget):
            # Evaluate the function with the current population
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(func_evals)
            # Update the adaptive function with the current population
            if np.any(best_func!= func(func_evals)):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Update the adaptive function with the best function
            if np.all(best_func == func(func_evals)):
                break
        return func_evals

# One-line description with the main idea
# AdaptiveHybridEvolutionaryAlgorithm: A hybrid algorithm that combines adaptive sampling and evolutionary optimization to solve black box optimization problems.