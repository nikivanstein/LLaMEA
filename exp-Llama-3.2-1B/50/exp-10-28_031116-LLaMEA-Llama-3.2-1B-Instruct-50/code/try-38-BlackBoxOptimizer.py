import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Evaluate the function for each solution in the population
        scores = []
        for solution in self.population:
            score = func(solution)
            scores.append(score)
            if len(scores) >= self.budget:
                break
        # Select the best solution based on the budget
        best_index = np.argmin(scores)
        best_solution = self.population[best_index]
        return best_solution

    def select_strategy(self):
        # Select a strategy based on the probability of success
        # For this example, we'll use a simple strategy that refines the solution
        # by changing the direction of the search
        if random.random() < 0.45:
            # Refine the solution by changing the direction of the search
            new_solution = best_solution + np.random.uniform(-1.0, 1.0, self.dim)
        else:
            # Use the current solution
            new_solution = best_solution
        return new_solution

    def mutate(self, solution):
        # Mutate the solution by changing one of its components
        # For this example, we'll use a simple mutation strategy that flips a random bit
        if random.random() < 0.01:
            # Flip a random bit
            index = random.randint(0, self.dim - 1)
            solution[index] = 1 - solution[index]
        return solution

    def evolve_population(self):
        # Evolve the population by applying the selection, mutation, and crossover strategies
        new_population = []
        for _ in range(self.population_size):
            # Select the best solution
            best_solution = self.__call__(self.func)
            # Select a strategy based on the probability of success
            strategy = self.select_strategy()
            # Mutate the solution
            mutated_solution = self.mutate(strategy)
            # Add the mutated solution to the new population
            new_population.append(mutated_solution)
        # Replace the old population with the new population
        self.population = new_population