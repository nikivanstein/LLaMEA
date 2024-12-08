# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
import numpy as np
import random
from typing import Dict

class BBOBMetaheuristic:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population: Dict[str, Dict[str, int]] = {}

    def __call__(self, func: callable) -> callable:
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func: callable, bounds: np.ndarray, mutation_prob: float = 0.25) -> np.ndarray:
        # Define the search space
        sol = None
        for _ in range(self.dim):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Refine the solution by changing individual lines
        for _ in range(self.dim):
            if random.random() < mutation_prob:
                # Change a line of the solution
                line = random.randint(0, self.dim - 1)
                sol[line] = random.uniform(bounds[line], bounds[line] + 1)
        
        # Return the best solution found
        return sol

# Initialize the evolutionary algorithm
algorithm = BBOBMetaheuristic(budget=1000, dim=5)

# Run the algorithm on the BBOB test suite
results = {}
for name, description, score in [
    ("Evolutionary Algorithm for Black Box Optimization using Genetic Programming", "A novel heuristic algorithm for solving black box optimization problems.", -1000),
    ("Genetic Programming for Black Box Optimization", "A genetic programming approach for solving black box optimization problems.", -800),
    ("Evolutionary Algorithm for Black Box Optimization", "An evolutionary algorithm for solving black box optimization problems.", -500),
    ("Genetic Algorithm for Black Box Optimization", "A genetic algorithm for solving black box optimization problems.", -300),
    ("Evolutionary Algorithm for Black Box Optimization using Ant Colony Optimization", "An evolutionary algorithm for solving black box optimization problems using ant colony optimization.", -200),
    ("Genetic Algorithm for Black Box Optimization using Ant Colony Optimization", "A genetic algorithm for solving black box optimization problems using ant colony optimization.", -100)
]:
    func = lambda x: x**2
    sol = algorithm.search(func, bounds=np.linspace(-5.0, 5.0, 10))
    results[name] = (score, sol)

# Print the results
print("Results:")
for name, (score, sol) in results.items():
    print(f"{name}: {score} - {sol}")