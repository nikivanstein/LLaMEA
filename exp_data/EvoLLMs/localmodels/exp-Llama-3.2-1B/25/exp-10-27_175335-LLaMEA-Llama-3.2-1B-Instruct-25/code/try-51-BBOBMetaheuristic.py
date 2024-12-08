import numpy as np
import random

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = []

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(100):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

    def mutate(self, individual):
        # Select two parents
        parent1, parent2 = random.sample(self.population, 2)
        
        # Select a random point in the search space
        crossover_point = random.randint(0, self.dim - 1)
        
        # Create a new offspring by combining the parents
        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        
        # Mutate the offspring by changing two random bits
        mutation_rate = 0.01
        mutated_offspring = offspring.copy()
        for _ in range(10):
            if random.random() < mutation_rate:
                mutated_offspring[random.randint(0, self.dim - 1)] ^= 1
        
        # Replace the parents with the mutated offspring
        self.population = [offspring] + [mutated_offspring[:]]

    def __repr__(self):
        return f"BBOBMetaheuristic(budget={self.budget}, dim={self.dim})"

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# BBOBMetaheuristic(budget=100, dim=10).search(lambda x: np.sin(x))  # Test the algorithm
# ```