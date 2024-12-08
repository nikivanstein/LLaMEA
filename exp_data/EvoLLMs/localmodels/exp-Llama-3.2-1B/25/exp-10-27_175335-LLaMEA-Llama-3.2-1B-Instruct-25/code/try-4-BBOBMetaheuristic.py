import numpy as np
import random

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.refining_strategies = {
            'random': self.random_refining_strategy,
            'bounded': self.bounded_refining_strategy
        }

    def random_refining_strategy(self, new_individual):
        # Refine the solution by changing 5% of the individual's elements
        updated_individual = new_individual.copy()
        for i in range(len(updated_individual)):
            if random.random() < 0.05:
                updated_individual[i] = random.uniform(updated_individual[i], updated_individual[i] + 1)
        return updated_individual

    def bounded_refining_strategy(self, new_individual):
        # Refine the solution by changing 10% of the individual's elements
        updated_individual = new_individual.copy()
        for i in range(len(updated_individual)):
            if random.random() < 0.1:
                updated_individual[i] = random.uniform(updated_individual[i], updated_individual[i] + 1)
        return updated_individual

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
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

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with Refining Strategies
# The algorithm uses a combination of random and bounded refining strategies to refine the solution
# The search space is defined by the bounds of the search space
# The algorithm evaluates the function at the current solution and updates it if it is better than the current best
# The solution is refined based on the probability of 0.25 for each refining strategy
# The algorithm uses 10 initializations and 10 iterations for each initialization