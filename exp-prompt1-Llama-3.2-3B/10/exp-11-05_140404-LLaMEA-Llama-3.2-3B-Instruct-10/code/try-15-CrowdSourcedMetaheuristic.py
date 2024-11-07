import numpy as np
from scipy.optimize import differential_evolution

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix
        self.population_size = int(np.ceil(self.budget * 0.2))  # Increase population size

    def __call__(self, func):
        for _ in range(self.budget):
            # Perform evolution strategy to update mean
            new_mean = self.mean + np.random.normal(0, 0.1, size=self.dim)  # Reduce noise
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space

            # Perform genetic drift to update covariance
            new_covariance = self.covariance + np.random.normal(0, 0.01, size=(self.dim, self.dim))  # Reduce noise
            new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix

            # Evaluate function at new mean
            f_new = func(new_mean)

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Store population
            self.population.append((self.mean, f_new))

            # Check for convergence
            if len(self.population) >= self.population_size:
                # Perform final optimization using differential evolution
                bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
                res = differential_evolution(func, bounds, x0=self.mean, seed=42)
                # Update mean and covariance
                self.mean = res.x
                self.covariance = np.eye(self.dim) * 1.0
                # Remove old population
                self.population = self.population[1:]

        # Print current best solution
        print(f"Current best solution: x = {self.mean}, f(x) = {func(self.mean)}")

    def get_best_solution(self):
        # Return best solution from population
        return min(self.population, key=lambda x: x[1])

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
crowd_sourced(func)
print("Best solution:", crowd_sourced.get_best_solution())

# Example usage with multiple dimensions
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=3)
crowd_sourced(func)
print("Best solution:", crowd_sourced.get_best_solution())