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

    # Probability-based refinement
    def refine_dphe(dphe, func):
        refined_individuals = []
        for i in range(len(dphe.fitness_history)):
            if np.random.rand() < 0.45:
                # Perturb the individual
                perturbed_individual = dphe.fitness_history[i] + np.random.uniform(-0.1, 0.1, size=len(dphe.fitness_history[i]))
                refined_individuals.append(perturbed_individual)
            else:
                refined_individuals.append(dphe.fitness_history[i])
        dphe.fitness_history = refined_individuals
        dphe.logger.update(dphe.fitness_history)
        return dphe

# Refine the DPHE algorithm
refined_dphe = refine_dphe(DPHE(budget=100, dim=10), func)

# Update the refined DPHE algorithm
def update_dphe(dphe, func):
    return refine_dphe(dphe, func)

# Update the DPHE algorithm
refined_dphe = update_dphe(DPHE(budget=100, dim=10), func)