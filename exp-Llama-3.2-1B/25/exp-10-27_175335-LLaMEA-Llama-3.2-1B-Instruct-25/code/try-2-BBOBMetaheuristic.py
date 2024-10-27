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

    def mutate(self, func, sol):
        # Refine the solution by changing 25% of the lines
        mutated_sol = sol.copy()
        for i in range(sol.size):
            if np.random.rand() < 0.25:
                mutated_sol[i] = np.random.uniform(-5.0, 5.0)
        
        # Evaluate the mutated function
        func_evals = self.func_evals
        self.func_evals += 1
        return func, mutated_sol

    def crossover(self, parent1, parent2):
        # Select the fittest parent and create a child
        fittest_parent = np.argmax(np.mean(parent1, axis=0))
        child = parent1.copy()
        
        # Refine the child by changing 50% of the lines
        for i in range(child.size):
            if np.random.rand() < 0.5:
                child[i] = np.random.uniform(-5.0, 5.0)
        
        # Evaluate the child
        func_evals = self.func_evals
        self.func_evals += 1
        return child, func, child

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 