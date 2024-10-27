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

# Probabilistic refinement
def refine_dphe(dphe, func, probability=0.45):
    new_individuals = []
    for _ in range(int(dphe.budget * probability)):
        new_individual = np.random.uniform(dphe.lower_bound, dphe.upper_bound, size=dphe.dim)
        new_individuals.append(new_individual)

    new_individuals = np.array(new_individuals)
    scores = np.array([func(individual) for individual in new_individuals])

    # Select the best 20% of new individuals
    best_individuals = np.random.choice(new_individuals, size=int(len(new_individuals) * 0.2), replace=False)

    # Refine the strategy by changing the individual lines of the selected individuals
    refined_individuals = best_individuals.copy()
    for i, individual in enumerate(refined_individuals):
        for j in range(dphe.dim):
            if np.random.rand() < probability:
                refined_individuals[i, j] = np.random.uniform(dphe.lower_bound, dphe.upper_bound)

    return refined_individuals

# Refine the DPHE algorithm
def refine_dphe_algorithm(dphe, func, probability=0.45):
    refined_individuals = refine_dphe(dphe, func, probability)
    dphe.f = refine_dphe.f
    return refined_individuals

# Refine the DPHE algorithm
refined_dphe_algorithm = refine_dphe_algorithm(DPHE(budget=100, dim=10), func)