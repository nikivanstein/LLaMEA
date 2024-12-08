import numpy as np
from scipy.optimize import differential_evolution

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix
        self.population_size = 10  # Increase population size for better exploration

    def __call__(self, func):
        for _ in range(self.budget):
            # Perform evolution strategy to update mean
            new_mean = self.mean + np.random.normal(0, 1.0, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space

            # Perform genetic drift to update covariance
            new_covariance = self.covariance + np.random.normal(0, 0.1, size=(self.dim, self.dim))
            new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix

            # Evaluate function at new mean
            f_new = func(new_mean)

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Store current best solution in population
            self.population.append((self.mean, f_new))

            # Perform crossover and mutation to generate new offspring
            offspring = []
            for _ in range(self.population_size - 1):
                parent1, parent2 = np.random.choice(self.population, size=2, replace=False)
                child = (parent1[0] + parent2[0]) / 2 + np.random.normal(0, 0.1, size=self.dim)
                child = np.clip(child, self.search_space[0], self.search_space[1])  # Clip values to search space
                f_child = func(child)
                offspring.append((child, f_child))

            # Update population with new offspring
            self.population = offspring

        # Perform final optimization using differential evolution
        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=self.mean, seed=42)
        print(f"Final best solution: x = {res.x}, f(x) = {res.fun}")

        # Return best solution from population
        return min(self.population, key=lambda x: x[1])

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
best_solution = crowd_sourced(func)
print(f"Best solution: x = {best_solution[0]}, f(x) = {best_solution[1]}")