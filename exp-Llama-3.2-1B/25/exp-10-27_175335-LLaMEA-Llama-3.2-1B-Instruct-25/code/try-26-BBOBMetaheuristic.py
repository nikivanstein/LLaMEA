import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func, bounds, mutation_rate, mutation_probability):
        # Define the search space
        bounds = np.linspace(bounds[0], bounds[1], self.dim, endpoint=False)
        
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

    def mutate(self, individual, mutation_rate):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        for _ in range(int(self.dim * mutation_rate)):
            mutated_individual[np.random.randint(0, self.dim)] += np.random.uniform(-1, 1)
        
        # Check if the mutation probability is reached
        if np.random.rand() < mutation_probability:
            # Apply the mutation
            mutated_individual[np.random.randint(0, self.dim)] -= np.random.uniform(-1, 1)
        
        return mutated_individual

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 