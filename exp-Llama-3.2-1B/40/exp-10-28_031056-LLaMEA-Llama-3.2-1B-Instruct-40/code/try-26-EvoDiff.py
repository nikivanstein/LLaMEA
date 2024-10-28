import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Tuple

class EvoDiff:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)

    def __call__(self, func: np.ndarray) -> np.ndarray:
        # Evaluate the black box function with the current population
        func_values = np.array([func(x) for x in self.population])

        # Select the fittest solutions
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = np.array([self.population[i] for i in fittest_indices])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection
            offspring = np.array([self.population[i] for i in np.argsort(mutated_parents)[::-1][:self.population_size]])

            # Replace the old population with the new one
            self.population = np.concatenate((self.population, mutated_parents), axis=0)
            self.population = np.concatenate((self.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

    def _tournament_selection(self, func: np.ndarray, func_values: np.ndarray, func_range: Tuple[float, float]) -> np.ndarray:
        # Select the fittest solutions using tournament selection
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]
        return np.array([func_values[i] for i in fittest_indices])

    def _mutation(self, mutated_parents: np.ndarray, mutation_rate: float) -> np.ndarray:
        # Perform mutation on the mutated parents
        mutated_parents = mutated_parents.copy()
        for _ in range(self.population_size):
            if np.random.rand() < mutation_rate:
                mutated_parents[_] += np.random.normal(0, 1, self.dim)
        return mutated_parents

    def _select_offspring(self, offspring: np.ndarray, func: np.ndarray, func_values: np.ndarray) -> np.ndarray:
        # Select the offspring using tournament selection
        return self._tournament_selection(func, func_values, (func_range[0], func_range[1]))