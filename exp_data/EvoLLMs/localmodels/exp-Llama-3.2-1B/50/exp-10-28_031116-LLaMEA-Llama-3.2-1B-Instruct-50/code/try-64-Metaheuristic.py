import random
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List

class Metaheuristic:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.population: List[Dict[str, float]] = []

    def __call__(self, func: callable) -> float:
        # Evaluate the black box function for the specified number of times
        scores = [func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)]
        # Select the best solution based on the budget
        best_idx = np.argmin(scores)
        best_func = scores[best_idx]
        # Optimize the best function using the selected strategy
        best_func = self.optimize_best_func(best_func, func, self.budget)
        return best_func

    def optimize_best_func(self, best_func: float, func: callable, budget: int) -> float:
        # Initialize the population with random solutions
        population = [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(100)]
        # Evolve the population using the selected strategy
        for _ in range(100):
            # Select the best solution based on the budget
            best_idx = np.argmin([func(x) for x in population])
            # Optimize the best function using the selected strategy
            best_func = self.optimize_best_func(best_func, func, budget)
            # Update the population
            population = self.update_population(population, best_func)
        # Return the best solution found
        return population[0][0]

    def update_population(self, population: List[Dict[str, float]], best_func: float) -> List[Dict[str, float]]:
        # Select the best solution based on the budget
        best_idx = np.argmin([func(x) for x in population])
        # Create a new population with the best solution
        new_population = [[x + random.uniform(-0.1, 0.1) for x in row] for row in population]
        # Combine the new population with the old population
        new_population = [dict(zip(self.dim, row)) for row in new_population]
        # Add the new population to the population
        self.population.append(new_population)
        return new_population

    def select_strategy(self, population: List[Dict[str, float]]) -> str:
        # Define the selection function based on the probability
        def selection_func(population: List[Dict[str, float]], best_func: float) -> float:
            # Calculate the fitness of each solution
            fitnesses = [best_func(x) for x in population]
            # Select the solutions with the highest fitness
            return np.argmax(fitnesses)
        # Select a random solution
        return random.choice(population)

    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        # Define the crossover function based on the probability
        def crossover_func(parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
            # Calculate the crossover point
            crossover_point = random.uniform(0, 0.5)
            # Create a new solution by combining the two parents
            child = {k: v for k, v in parent1.items() if k not in parent2 or v > parent2[k]}
            # Add the crossover point to the child solution
            child[crossover_point] = parent2[crossover_point]
            return child
        # Crossover the two parents
        return crossover_func(parent1, parent2)

    def mutate(self, solution: Dict[str, float]) -> Dict[str, float]:
        # Define the mutation function based on the probability
        def mutation_func(solution: Dict[str, float]) -> Dict[str, float]:
            # Calculate the mutation point
            mutation_point = random.uniform(0, 0.1)
            # Create a new solution by modifying the mutation point
            solution[mutation_point] = random.uniform(-0.1, 0.1)
            return solution
        # Mutate the solution
        return mutation_func(solution)