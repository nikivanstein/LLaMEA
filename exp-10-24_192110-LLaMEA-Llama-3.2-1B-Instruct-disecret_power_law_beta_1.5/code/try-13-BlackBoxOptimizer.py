import numpy as np
from scipy.optimize import differential_evolution
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = deque(maxlen=1000)

    def __call__(self, func):
        # Evaluate the black box function
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=[np.random.uniform(-5.0, 5.0) for _ in range(self.dim)])
        # Optimize the function
        self.population.append((res.x, res.fun))
        # Update the best solution
        if len(self.population) > self.budget:
            best_sol = max(self.population, key=lambda x: x[1])
            if best_sol[1] > 0:
                func = lambda x: -x[1]  # Minimize the negative of the function
            elif best_sol[1] < 0:
                func = lambda x: -x[1]  # Maximize the negative of the function
        return func

    def select_solution(self):
        # Select the best solution
        best_sol = max(self.population, key=lambda x: x[1])
        if best_sol[1] > 0:
            best_sol = (best_sol[0], -best_sol[1])  # Minimize the negative of the function
        elif best_sol[1] < 0:
            best_sol = (best_sol[0], -best_sol[1])  # Maximize the negative of the function
        return best_sol

    def run(self):
        # Run the optimization algorithm
        best_sol = self.select_solution()
        best_score = -best_sol[1]
        print(f"Best solution: {best_sol} with score: {best_score}")
        func = lambda x: best_sol[1]
        return func

# Test the algorithm
optimizer = BlackBoxOptimizer(budget=100, dim=5)
func = optimizer(__call__)
print(func())