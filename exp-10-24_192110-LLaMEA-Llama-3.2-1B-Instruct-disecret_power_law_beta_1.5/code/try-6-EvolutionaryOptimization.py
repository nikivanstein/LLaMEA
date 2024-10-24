import numpy as np

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitnesses = []

    def __call__(self, func):
        # Define the search space
        bounds = [-5.0, 5.0]
        # Initialize the population with random solutions
        self.population = [[np.random.uniform(bounds[0], bounds[1], self.dim) for _ in range(self.dim)] for _ in range(100)]
        # Evaluate the function for each solution
        for solution in self.population:
            func(solution)
            # Update the fitness score
            self.fitnesses.append(solution)
        # Select the fittest solutions
        self.population = self.select_fittest(self.fitnesses, 10)
        # Calculate the fitness scores for each solution
        self.fitnesses = [np.mean([np.abs(solution - func(solution)) for solution in self.population]) for solution in self.population]
        # Update the population
        self.population = self.population[:self.budget]

    def select_fittest(self, fitnesses, k):
        # Select the k fittest solutions based on their fitness scores
        return sorted(zip(fitnesses, self.population), key=lambda x: x[0], reverse=True)[:k]

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 