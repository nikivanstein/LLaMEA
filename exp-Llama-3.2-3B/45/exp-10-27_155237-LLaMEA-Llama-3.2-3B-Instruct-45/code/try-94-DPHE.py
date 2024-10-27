import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            return res.x
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe = DPHE(budget=100, dim=10)

    # Optimize the function
    result = dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")

# Refine DPHE strategy with probability 0.45
def refine_dphe_dphe(dphe, func):
    # Calculate the probability of mutation
    prob_mutate = 0.45

    # Initialize the new population
    new_population = []

    # Iterate over the current population
    for i in range(len(dphe.fitness_history)):
        # Select the current individual
        individual = dphe.fitness_history[i]

        # Generate a new individual by perturbing the current individual
        new_individual = individual + np.random.normal(0, 0.1, size=dphe.dim)

        # Check if the new individual is within the bounds
        if np.all(new_individual >= dphe.lower_bound) and np.all(new_individual <= dphe.upper_bound):
            # Add the new individual to the new population
            new_population.append(new_individual)
        else:
            # If the new individual is out of bounds, try again
            new_individual = individual + np.random.normal(0, 0.1, size=dphe.dim)
            if np.all(new_individual >= dphe.lower_bound) and np.all(new_individual <= dphe.upper_bound):
                new_population.append(new_individual)

    # Update the fitness history
    dphe.fitness_history = new_population

    # Return the updated individual
    return dphe.fitness_history[np.random.randint(0, len(dphe.fitness_history))]

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe = DPHE(budget=100, dim=10)

    # Optimize the function
    for i in range(10):
        # Refine the DPHE strategy
        result = refine_dphe_dphe(dphe, func)
        # Optimize the function
        result = dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")