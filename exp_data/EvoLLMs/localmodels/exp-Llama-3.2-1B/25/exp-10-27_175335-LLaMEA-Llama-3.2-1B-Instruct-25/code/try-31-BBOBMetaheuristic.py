import numpy as np
import random

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

    def mutate(self, sol):
        # Randomly select an individual to mutate
        idx = random.randint(0, self.dim - 1)
        
        # Create a new individual by refining the mutated line
        new_sol = sol.copy()
        new_sol[idx] += random.uniform(-1, 1)
        
        # Check if the new individual is within the bounds
        if new_sol[idx] < -5.0 or new_sol[idx] > 5.0:
            raise ValueError("Mutated individual is out of bounds")
        
        # Update the solution
        sol = new_sol
        
        # Evaluate the function at the new solution
        func_sol = self.__call__(func, sol)
        
        # Check if the new solution is better than the current best
        if func_sol < self.__call__(func, sol):
            # Update the solution
            sol = new_sol
        
        # Return the mutated solution
        return sol

    def crossover(self, sol1, sol2):
        # Randomly select a crossover point
        idx = random.randint(0, self.dim - 1)
        
        # Create a new individual by combining the two parents
        new_sol = np.concatenate((sol1[:idx], sol2[idx:]))
        
        # Check if the new individual is within the bounds
        if new_sol[idx] < -5.0 or new_sol[idx] > 5.0:
            raise ValueError("Crossover individual is out of bounds")
        
        # Evaluate the function at the new solution
        func_sol = self.__call__(func, new_sol)
        
        # Check if the new solution is better than the current best
        if func_sol < self.__call__(func, new_sol):
            # Update the solution
            new_sol = new_sol
        
        # Return the new solution
        return new_sol

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# class BBOBMetaheuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0

#     def __call__(self, func):
#         # Check if the function can be evaluated within the budget
#         if self.func_evals >= self.budget:
#             raise ValueError("Not enough evaluations left to optimize the function")

#         # Evaluate the function within the budget
#         func_evals = self.func_evals
#         self.func_evals += 1
#         return func

#     def search(self, func):
#         # Define the search space
#         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
#         # Initialize the solution
#         sol = None
        
#         # Try different initializations
#         for _ in range(10):
#             # Randomly initialize the solution
#             sol = np.random.uniform(bounds, size=self.dim)
            
#             # Evaluate the function at the solution
#             func_sol = self.__call__(func, sol)
            
#             # Check if the solution is better than the current best
#             if func_sol < self.__call__(func, sol):
#                 # Update the solution
#                 sol = sol
        
#         # Return the best solution found
#         return sol

#     def mutate(self, sol):
#         # Randomly select an individual to mutate
#         idx = random.randint(0, self.dim - 1)
        
#         # Create a new individual by refining the mutated line
#         new_sol = sol.copy()
#         new_sol[idx] += random.uniform(-1, 1)
        
#         # Check if the new individual is within the bounds
#         if new_sol[idx] < -5.0 or new_sol[idx] > 5.0:
#             raise ValueError("Mutated individual is out of bounds")
        
#         # Update the solution
#         sol = new_sol
        
#         # Evaluate the function at the new solution
#         func_sol = self.__call__(func, sol)
        
#         # Check if the new solution is better than the current best
#         if func_sol < self.__call__(func, sol):
#             # Update the solution
#             sol = new_sol
        
#         # Return the mutated solution
#         return sol

#     def crossover(self, sol1, sol2):
#         # Randomly select a crossover point
#         idx = random.randint(0, self.dim - 1)
        
#         # Create a new individual by combining the two parents
#         new_sol = np.concatenate((sol1[:idx], sol2[idx:]))
        
#         # Check if the new individual is within the bounds
#         if new_sol[idx] < -5.0 or new_sol[idx] > 5.0:
#             raise ValueError("Crossover individual is out of bounds")
        
#         # Evaluate the function at the new solution
#         func_sol = self.__call__(func, new_sol)
        
#         # Check if the new solution is better than the current best
#         if func_sol < self.__call__(func, new_sol):
#             # Update the solution
#             new_sol = new_sol
        
#         # Return the new solution
#         return new_sol

# def main():
#     # Initialize the algorithm
#     algorithm = BBOBMetaheuristic(1000, 10)
        
#     # Optimize a function
#     func = lambda x: x**2
#     best_sol = algorithm.search(func)
#     best_func = algorithm.__call__(func, best_sol)
        
#     # Print the results
#     print(f"Best solution: {best_sol}")
#     print(f"Best function value: {best_func}")
        
#     # Plot the results
#     import matplotlib.pyplot as plt
#     plt.plot([best_sol, best_func], [best_func, best_func], color='red')
#     plt.show()

# if __name__ == "__main__":
#     main()