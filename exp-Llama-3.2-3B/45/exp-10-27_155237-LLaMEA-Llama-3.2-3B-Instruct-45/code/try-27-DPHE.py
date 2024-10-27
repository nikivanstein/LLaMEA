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
    def refine_solution(solution, prob=0.45):
        if np.random.rand() < prob:
            # Randomly select two different dimensions
            idx1, idx2 = np.random.choice(range(solution.shape[0]), size=2, replace=False)
            # Perturb the selected dimensions
            perturbation = np.random.uniform(-1, 1, size=2)
            # Create a new solution by adding the perturbation to the original solution
            new_solution = solution.copy()
            new_solution[idx1] += perturbation[0]
            new_solution[idx2] += perturbation[1]
            return new_solution
        else:
            return solution

    # Refine the solution using the probabilistic refinement function
    refined_solution = refine_solution(result)

    # Print the refined solution
    print("Refined solution:", refined_solution)