import numpy as np
from scipy.optimize import differential_evolution

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

class EvolutionaryBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim, mutation_prob=0.25):
        super().__init__(budget, dim)
        self.mutation_prob = mutation_prob

    def __call__(self, func):
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
                if np.random.rand() < self.mutation_prob:
                    # Perform mutation
                    sol = self.mutation(sol)
        
        # Return the best solution found
        return sol

    def mutation(self, sol):
        # Generate a new solution by refining the current one
        new_sol = sol.copy()
        for i in range(self.dim):
            # Refine the solution using a probability
            if np.random.rand() < self.mutation_prob:
                # Randomly change the value in the solution
                new_sol[i] += np.random.uniform(-1, 1)
        
        # Check if the new solution is better than the current best
        if self.__call__(func, new_sol) < self.__call__(func, sol):
            # Update the solution
            new_sol = new_sol.copy()
            for i in range(self.dim):
                # Refine the solution using a probability
                if np.random.rand() < self.mutation_prob:
                    # Randomly change the value in the solution
                    new_sol[i] += np.random.uniform(-1, 1)
        
        # Return the new solution
        return new_sol

# Description: Evolutionary BBOB Metaheuristic
# Code: 