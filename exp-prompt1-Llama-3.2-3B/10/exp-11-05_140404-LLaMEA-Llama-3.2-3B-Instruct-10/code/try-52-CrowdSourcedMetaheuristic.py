import numpy as np
from scipy.optimize import differential_evolution

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix
        self.num_objectives = 1  # Number of objectives to optimize

    def __call__(self, funcs):
        for _ in range(self.budget):
            # Perform evolution strategy to update mean
            new_mean = self.mean + np.random.normal(0, 1.0, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space

            # Perform genetic drift to update covariance
            new_covariance = self.covariance + np.random.normal(0, 0.1, size=(self.dim, self.dim))
            new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix

            # Evaluate objectives at new mean
            f_new = np.array([func(new_mean) for func in funcs])

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Print current best solution
            print(f"Current best solution: x = {self.mean}, f(x) = {f_new}")

        # Perform multi-objective optimization using Pareto-based differential evolution
        def multi_objective_func(x):
            f_values = np.array([func(x) for func in funcs])
            return f_values

        def pareto_front(x):
            f_values = np.array([func(x) for func in funcs])
            return f_values

        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
        res = differential_evolution(multi_objective_func, bounds, x0=self.mean, seed=42, population_size=50)
        print(f"Final Pareto front: x = {res.x}, f(x) = {res.fun}")

# Example usage
def func1(x):
    return x[0]**2 + x[1]**2
def func2(x):
    return x[0]**3 + x[1]**3

funcs = [func1, func2]
crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
crowd_sourced(funcs)